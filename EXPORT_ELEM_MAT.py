import re
import csv
from pathlib import Path


def parse_fixed_width_ints(line: str, width: int = 10):
    """
    Parse fixed-width integer fields used by ANSYS CDB, e.g. (19i10).
    Falls back to regex parsing if fixed-width parsing fails.
    """
    values = []

    # Try fixed-width first
    for i in range(0, len(line), width):
        chunk = line[i:i + width].strip()
        if not chunk:
            continue
        try:
            values.append(int(chunk))
        except ValueError:
            values = []
            break

    if values:
        return values

    # Fallback: whitespace / regex integer parsing
    return [int(x) for x in re.findall(r"[-+]?\d+", line)]


def extract_material_lines(cdb_text: str):
    """
    Extract APDL material definition lines from mapped CDB.
    Keeps common material commands.
    """
    material_prefixes = (
        "MP,",
        "MPDATA,",
        "MPTEMP,",
        "TB,",
        "TBDATA,",
        "TBTEMP,",
    )

    lines = []
    for raw in cdb_text.splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith(material_prefixes):
            lines.append(line)

    return lines


def extract_elem_mat_from_eblock(cdb_text: str):
    """
    Parse EBLOCK sections and extract element_id -> material_id.

    For ANSYS EBLOCK SOLID format, the first records are typically:
    MAT, TYPE, REAL, SECNUM, ESYS, ..., NNODES, ELEM_ID, NODE1, ...

    Usually:
    material_id = record[0]
    node_count  = record[8]
    element_id  = record[9]
    """
    elem_mat = {}

    lines = cdb_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip().upper()

        if line.startswith("EBLOCK"):
            # Next line is usually the format line, e.g. (19i10)
            i += 1

            width = 10
            if i < len(lines):
                fmt_line = lines[i].strip().lower()
                m = re.search(r"i(\d+)", fmt_line)
                if m:
                    width = int(m.group(1))

            i += 1

            buffer = []

            while i < len(lines):
                raw = lines[i]
                stripped = raw.strip()

                # End of EBLOCK is often -1
                if stripped.startswith("-1"):
                    break

                nums = parse_fixed_width_ints(raw.rstrip("\n"), width=width)
                if not nums:
                    i += 1
                    continue

                buffer.extend(nums)

                # Parse as many complete element records as possible
                while len(buffer) >= 10:
                    mat_id = buffer[0]
                    try:
                        node_count = buffer[8]
                        elem_id = buffer[9]
                    except IndexError:
                        break

                    # Basic sanity check
                    if node_count <= 0 or node_count > 50 or elem_id <= 0:
                        # Not a standard solid element record; discard one int and continue
                        buffer.pop(0)
                        continue

                    record_len = 10 + node_count
                    if len(buffer) < record_len:
                        break

                    elem_mat[elem_id] = mat_id

                    # Remove this element record from buffer
                    buffer = buffer[record_len:]

                i += 1

        i += 1

    return elem_mat


def write_material_defs(material_lines, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("/PREP7\n")
        f.write("! Material definitions extracted from Bonemat mapped CDB\n")
        for line in material_lines:
            f.write(line + "\n")
        f.write("ALLSEL,ALL\n")


def write_elem_mat_csv(elem_mat, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ELEM", "MAT"])
        for elem_id in sorted(elem_mat):
            writer.writerow([elem_id, elem_mat[elem_id]])


def write_apply_elem_mat_macro(elem_mat, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("/PREP7\n")
        f.write("! Element material assignment extracted from Bonemat mapped CDB\n")
        for elem_id in sorted(elem_mat):
            mat_id = elem_mat[elem_id]
            f.write(f"EMODIF,{elem_id},MAT,{mat_id}\n")
        f.write("ALLSEL,ALL\n")


def main():
    # 修改这里：填你的 Bonemat mapped CDB 文件名
    mapped_cdb = Path(r"/mnt/PhD/pythonproject/PHD_PROGRAM/GCN_Stress/vertebrae_to_APDL.cdb")

    if not mapped_cdb.exists():
        raise FileNotFoundError(f"Cannot find CDB file: {mapped_cdb}")

    cdb_text = mapped_cdb.read_text(errors="ignore")

    material_lines = extract_material_lines(cdb_text)
    elem_mat = extract_elem_mat_from_eblock(cdb_text)

    out_dir = mapped_cdb.parent

    write_material_defs(material_lines, out_dir / "material_defs.mac")
    write_elem_mat_csv(elem_mat, out_dir / "elem_mat.csv")
    write_apply_elem_mat_macro(elem_mat, out_dir / "apply_elem_mat.mac")

    print("Done.")
    print(f"Material definition lines: {len(material_lines)}")
    print(f"Element material assignments: {len(elem_mat)}")
    print(f"Output folder: {out_dir}")


if __name__ == "__main__":
    main()