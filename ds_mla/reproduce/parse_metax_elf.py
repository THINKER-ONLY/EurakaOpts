#!/usr/bin/env python3
import argparse, base64, hashlib, json, os, struct, sys
SHT_SYMTAB, SHT_NOTE = 2, 7
KERNEL_KEYS = (".name", ".mtreg_count", ".streg_count", ".private_memory_size", ".share_memory_size", ".max_block_size", ".kd_symbol", ".args")
class MPError(Exception): pass
class MP:
    def __init__(self, data, base): self.data, self.pos, self.base = data, 0, base
    def fail(self, msg): raise MPError(f"{msg} at descriptor offset {self.pos} (file offset {self.base+self.pos})")
    def take(self, n):
        if n < 0 or self.pos+n > len(self.data): self.fail(f"truncated MessagePack: need {n} bytes, {len(self.data)-self.pos} remain")
        b = self.data[self.pos:self.pos+n]; self.pos += n; return b
    def num(self, n): return int.from_bytes(self.take(n), "big")
    def obj(self):
        p, b = self.pos, self.num(1)
        if b <= 0x7f: return b
        if b >= 0xe0: return b-256
        if 0x80 <= b <= 0x8f: return self.mapping(b&15)
        if 0x90 <= b <= 0x9f: return self.array(b&15)
        if 0xa0 <= b <= 0xbf: return self.string(b&31)
        if b in (0xc0, 0xc2, 0xc3): return {0xc0: None, 0xc2: False, 0xc3: True}[b]
        if b in (0xc4, 0xc5, 0xc6): return self.binary(self.num(1 << (b-0xc4)))
        if b in (0xca, 0xcb): return struct.unpack(">f" if b == 0xca else ">d", self.take(4 if b == 0xca else 8))[0]
        if b in (0xcc, 0xcd, 0xce, 0xcf): return self.num(1 << (b-0xcc))
        if b in (0xd0, 0xd1, 0xd2, 0xd3): return int.from_bytes(self.take(1 << (b-0xd0)), "big", signed=True)
        if b in (0xd9, 0xda, 0xdb): return self.string(self.num(1 << (b-0xd9)))
        if b in (0xdc, 0xdd): return self.array(self.num(2 if b == 0xdc else 4))
        if b in (0xde, 0xdf): return self.mapping(self.num(2 if b == 0xde else 4))
        self.pos = p; self.fail(f"unknown MessagePack type 0x{b:02x}")
    def string(self, n):
        start = self.pos; raw = self.take(n)
        try: return raw.decode("utf-8")
        except UnicodeDecodeError as e:
            self.pos = start; self.fail(f"invalid UTF-8 string ({e})")
    def binary(self, n): return {"$binary_base64": base64.b64encode(self.take(n)).decode("ascii")}
    def array(self, n):
        if n > len(self.data)-self.pos: self.fail(f"array length {n} exceeds remaining bytes")
        return [self.obj() for _ in range(n)]
    def mapping(self, n):
        if n > len(self.data)-self.pos: self.fail(f"map length {n} exceeds remaining bytes")
        out = {}
        for _ in range(n):
            kpos = self.pos; k = self.obj()
            if isinstance(k, (dict, list)): self.pos = kpos; self.fail("unhashable map key")
            out[k] = self.obj()
        return out
def need(data, off, size, what):
    if off < 0 or size < 0 or off+size > len(data): raise ValueError(f"{what} out of bounds: offset={off} size={size} file_size={len(data)}")
def parse_elf(data):
    if len(data) < 64 or data[:4] != b"\x7fELF": raise ValueError("not an ELF file")
    if data[4] != 2 or data[5] != 1: raise ValueError("only ELF64 little-endian is supported")
    shoff = struct.unpack_from("<Q", data, 40)[0]; shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 58)
    if shentsize < 64: raise ValueError(f"e_shentsize={shentsize} is smaller than 64")
    need(data, shoff, 64, "section header 0")
    def raw(i):
        o = shoff+i*shentsize; need(data, o, 64, f"section header {i}")
        return struct.unpack_from("<IIQQQQIIQQ", data, o)
    if shnum == 0: shnum = raw(0)[5]
    if shnum < 1 or shnum > (len(data)-shoff)//shentsize: raise ValueError(f"invalid section count {shnum}")
    if shstrndx == 0xffff: shstrndx = raw(0)[6]
    sections = [raw(i) for i in range(shnum)]
    if shstrndx == 0: names = [""]*shnum
    else:
        if shstrndx >= shnum: raise ValueError(f"e_shstrndx={shstrndx} is out of range")
        so, ss = sections[shstrndx][4], sections[shstrndx][5]; need(data, so, ss, "section name string table"); blob = data[so:so+ss]; names = []
        for s in sections:
            no = s[0]
            if no >= len(blob) or blob.find(b"\0", no) < 0: raise ValueError(f"invalid section name offset {no}")
            z = blob.find(b"\0", no)
            try: names.append(blob[no:z].decode("utf-8"))
            except UnicodeDecodeError: raise ValueError(f"section name at offset {no} is not UTF-8")
    if len(names) != shnum: raise ValueError("section name table is incomplete")
    return {"sections": sections, "names": names, "shoff": shoff, "shentsize": shentsize, "shnum": shnum, "shstrndx": shstrndx}
def parse_notes(data, sections, names):
    notes = []
    for i, s in enumerate(sections):
        if s[1] != SHT_NOTE: continue
        off, size = s[4], s[5]; need(data, off, size, f"SHT_NOTE section {i}"); pos, end = off, off+size
        while pos < end:
            if end-pos < 12: raise ValueError(f"truncated note header at file offset {pos}")
            namesz, descsz, typ = struct.unpack_from("<III", data, pos); no = pos+12
            ds = (no+namesz+3) & ~3; de = ds+descsz; nxt = (de+3) & ~3
            if no+namesz > end or de > end or nxt > end: raise ValueError(f"note at file offset {pos} exceeds SHT_NOTE bounds")
            owner = data[no:no+namesz].rstrip(b"\0").decode("utf-8", "replace")
            notes.append({"section": names[i], "note_offset": pos, "owner": owner, "type": typ, "descriptor_offset": ds, "descriptor_length": descsz, "descriptor_sha256": hashlib.sha256(data[ds:de]).hexdigest(), "_descriptor": data[ds:de]}); pos = nxt
    return notes
def parse_symtab(data, sections, names):
    funcs, errors = [], []
    for i, s in enumerate(sections):
        if s[1] != SHT_SYMTAB: continue
        off, size, link, ent = s[4], s[5], s[6], s[9] or 24
        if ent < 24: errors.append(f"SHT_SYMTAB {i} entry size {ent} < 24"); continue
        if link >= len(sections): errors.append(f"SHT_SYMTAB {i} invalid sh_link {link}"); continue
        try:
            need(data, off, size, f"SHT_SYMTAB {i}"); st = sections[link]; need(data, st[4], st[5], f"symbol string table for section {i}")
        except ValueError as e:
            errors.append(str(e)); continue
        strtab = data[st[4]:st[4]+st[5]]
        for j in range(size//ent):
            nameoff, info, other, shndx, value, sz = struct.unpack_from("<IBBHQQ", data, off+j*ent)
            if info & 15 != 2 or shndx == 0: continue
            if nameoff >= len(strtab) or strtab.find(b"\0", nameoff) < 0: errors.append(f"symbol {j} in section {i} has invalid name offset"); continue
            try: name = strtab[nameoff:strtab.find(b"\0", nameoff)].decode("utf-8")
            except UnicodeDecodeError: errors.append(f"symbol {j} in section {i} has invalid UTF-8 name"); continue
            funcs.append({"name": name, "value": value, "size": sz, "shndx": shndx, "symtab": names[i]})
    return funcs, errors
def kernels(meta, funcs):
    ks = meta.get("macahca.kernels") if isinstance(meta, dict) else None
    if isinstance(ks, dict): ks = list(ks.values())
    if not isinstance(ks, list): return []
    sizes = {}
    for f in funcs: sizes.setdefault(f["name"], f["size"])
    out = []
    for k in ks:
        if not isinstance(k, dict): continue
        row = {key: k[key] for key in KERNEL_KEYS if key in k}
        target = k.get(".kd_symbol") if isinstance(k.get(".kd_symbol"), str) else k.get(".name")
        row["function_byte_size"] = sizes.get(target) if isinstance(target, str) else None
        out.append(row)
    return out
def process_file(path, name):
    rec = {"file": name, "errors": []}
    try:
        with open(path, "rb") as f: data = f.read()
    except Exception as e:
        rec["errors"].append(f"read: {e}"); return rec
    rec.update(size=len(data), sha256=hashlib.sha256(data).hexdigest())
    try:
        elf = parse_elf(data); sections, names = elf["sections"], elf["names"]
        rec["elf"] = {k: elf[k] for k in ("shoff", "shentsize", "shnum", "shstrndx")}
        rec["sections"] = [{"index": i, "name": names[i], "type": s[1], "offset": s[4], "size": s[5]} for i, s in enumerate(sections)]
        rec["text_sections"] = [x for x in rec["sections"] if x["name"] == ".text"]
        rec["text_section_size"] = sum(x["size"] for x in rec["text_sections"])
    except Exception as e:
        rec["errors"].append(f"ELF: {e}"); return rec
    funcs, serrors = [], []
    try: funcs, serrors = parse_symtab(data, sections, names)
    except Exception as e: serrors.append(f"symtab: {e}")
    rec["functions"] = funcs; rec["function_count"] = len(funcs); rec["errors"].extend(serrors)
    try: notes = parse_notes(data, sections, names)
    except Exception as e:
        rec["errors"].append(f"notes: {e}"); notes = []
    descriptors, metas = [], []
    for n in notes:
        d = {k: v for k, v in n.items() if not k.startswith("_")}
        if n["type"] == 0x30 and "metax" in n["owner"].lower():
            try:
                mp = MP(n["_descriptor"], n["descriptor_offset"]); meta = mp.obj()
                d.update(metadata=meta, trailing_bytes=len(n["_descriptor"])-mp.pos, decoded_length=mp.pos)
                metas.append(meta)
            except MPError as e:
                d["error"] = str(e); rec["errors"].append(f"descriptor@{n['descriptor_offset']}: {e}")
        descriptors.append(d)
    rec["descriptors"] = descriptors; rec["descriptor_sha256"] = [d["descriptor_sha256"] for d in descriptors]
    if metas: rec["metadata"] = metas[0]; rec["macahca.version"] = metas[0].get("macahca.version") if isinstance(metas[0], dict) else None
    else: rec["metadata"] = None; rec["errors"].append("no MetaX type 0x30 MessagePack descriptor found")
    rec["kernels"] = kernels(metas[0], funcs) if metas and isinstance(metas[0], dict) else []
    return rec
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True); ap.add_argument("--output", required=True)
    a = ap.parse_args(); out, inp = os.path.abspath(a.output), os.path.abspath(a.input_dir); errors, results = [], []
    if out == inp or out.startswith(inp+os.sep): print("refusing to write inside --input-dir", file=sys.stderr); return 2
    try: entries = sorted((e for e in os.scandir(a.input_dir) if e.is_file() and e.name.endswith(".mcbin")), key=lambda e: e.name)
    except OSError as e: entries = []; errors.append(f"scan: {e}")
    for e in entries: results.append(process_file(e.path, e.name))
    if not entries and not errors: errors.append("no .mcbin files")
    failed = sum(bool(r.get("errors")) for r in results)
    summary = {"file_count": len(results), "failed_files": failed, "ok": not errors and failed == 0, "total_kernels": sum(len(r.get("kernels", [])) for r in results), "total_functions": sum(r.get("function_count", 0) for r in results)}
    payload = {"hardware_counters_collected": False, "input_dir": inp, "summary": summary, "errors": errors, "files": results}
    try:
        with open(out, "x", encoding="utf-8") as fh: json.dump(payload, fh, indent=2, ensure_ascii=False)
    except OSError as e: print(f"cannot exclusively create {out}: {e}", file=sys.stderr); return 2
    for r in results:
        print(f"{r['file']}: sha256={r.get('sha256','-')} text_size={r.get('text_section_size',0)} functions={r.get('function_count',0)} kernels={len(r.get('kernels',[]))} errors={len(r.get('errors',[]))}")
        for k in r.get("kernels", []): print(f"  {k.get('.name')}: mtreg={k.get('.mtreg_count')} streg={k.get('.streg_count')} private={k.get('.private_memory_size')} shared={k.get('.share_memory_size')} max_block={k.get('.max_block_size')} kd_symbol={k.get('.kd_symbol')} function_bytes={k.get('function_byte_size')}")
    print(f"summary: files={len(results)} failed={failed} kernels={summary['total_kernels']} functions={summary['total_functions']} output={out}")
    for e in errors: print(f"error: {e}", file=sys.stderr)
    return 0 if summary["ok"] else 1
if __name__ == "__main__": sys.exit(main())
