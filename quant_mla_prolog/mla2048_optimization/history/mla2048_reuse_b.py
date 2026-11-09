from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
start = source.index("class _O11MlaWkbTokenMajorLargeUnit:")
end = source.index("class _MlaPrefillQbUnit:", start)
body = source[start:end]
loop_start = body.index("        for m_tile in range(m_tiles):")
operand_start = body.index("            m_start = m_tile * self.bm", loop_start)
column_start = body.index("            for n_tile in range_constexpr(4):", operand_start)
result_start = body.index("                c_l0_write = self.c_l0.produce()", column_start)
activation = body[operand_start:column_start]
weight = body[column_start:result_start].split("\n", 1)[1]
weight = "\n".join(line[4:] if line.startswith("    ") else line for line in weight.split("\n"))
result = body[result_start:].rstrip()
body = body[:loop_start] + "        for n_tile in range_constexpr(4):\n" + weight
body += "            b_l0_read = self.b_l0.consume()\n"
body += "            for m_tile in range(m_tiles):\n"
body += "\n".join("    " + line if line else line for line in activation.split("\n"))
body += result.replace("self.b_l0.consume()", "b_l0_read") + "\n\n\n"
source = source[:start] + body + source[end:]
compile(source, str(path), "exec")
path.write_text(source)
