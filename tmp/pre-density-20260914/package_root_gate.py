"""按主代理指定的完整提交归档推理门控代码, 不混入当前性能实现树."""
from pathlib import Path
import base64,subprocess,io,tarfile,json,hashlib
task=Path(__file__).resolve().parent
commit='0a9c849'
original=subprocess.check_output(['git','archive','--format=tar.gz',commit],cwd=task.parents[1])
converted=[]
normalized=io.BytesIO()
with tarfile.open(fileobj=io.BytesIO(original),mode='r:gz') as source, tarfile.open(fileobj=normalized,mode='w:gz') as target:
 for member in source.getmembers():
  if not member.isfile(): target.addfile(member);continue
  content=source.extractfile(member).read()
  if member.name.endswith('.sh') and b'\r\n' in content:
   content=content.replace(b'\r\n',b'\n');converted.append(member.name)
  member.size=len(content);target.addfile(member,io.BytesIO(content))
 manifest=json.dumps(dict(commit=commit,original_archive_sha256=hashlib.sha256(original).hexdigest(),shell_files_normalized_to_lf=converted),indent=2).encode()
 member=tarfile.TarInfo('root_gate_archive_manifest.json');member.size=len(manifest);target.addfile(member,io.BytesIO(manifest))
payload=normalized.getvalue()
remote='/storage/penghongen/tmp/pocketxmol_density_20260914'
script=f'''#!/usr/bin/env bash
set -euo pipefail
mkdir -p '{remote}/sampling_gate_{commit}/source'
base64 -d <<'PXM_ROOT_GATE' | tar -xz -C '{remote}/sampling_gate_{commit}/source'
{base64.b64encode(payload).decode()}
PXM_ROOT_GATE
printf 'ROOT_GATE_UPLOAD commit={commit} bytes={len(payload)}\\n'
'''
(task/'upload_root_gate.sh').write_text(script,encoding='utf-8',newline='\n')
print('archive_bytes',len(payload))
