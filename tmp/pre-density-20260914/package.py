"""打包本次隔离代码并产生可经SSH标准输入上传的脚本。"""
from pathlib import Path
import base64,tarfile,io
root=Path(__file__).resolve().parents[2]
task=root/'tmp/pre-density-20260914'
archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w:gz') as tar:
 for directory in ('models','docking','scripts','utils','configs','process','datasets','训练与运行/runtime'):
  for path in (root/directory).rglob('*'):
   if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc',):
    if path.suffix=='.sh':
     data=path.read_bytes().replace(b'\r\n',b'\n');info=tar.gettarinfo(str(path),arcname=str(path.relative_to(root)));info.size=len(data);tar.addfile(info,io.BytesIO(data))
    else: tar.add(path,arcname=path.relative_to(root))
 for path in task.iterdir():
  if path.suffix in ('.py','.sh') and path.name not in ('upload.sh','upload_root_gate.sh'):
   data=path.read_bytes().replace(b'\r\n',b'\n');info=tar.gettarinfo(str(path),arcname=str(path.relative_to(root)));info.size=len(data);tar.addfile(info,io.BytesIO(data))
 tar.add(root/'tests/test_density.py',arcname='tests/test_density.py')
 for name in ('raunet.py','attention_3d_rope.py'):
  tar.add(Path('C:/Users/15919/Desktop/Pocket_Plus/src/model')/name,arcname='tmp/pre-density-20260914/reference/'+name)
payload=base64.b64encode(archive.getvalue()).decode()
remote='/storage/penghongen/tmp/pocketxmol_density_20260914'
script=f'''#!/usr/bin/env bash
set -euo pipefail
mkdir -p '{remote}/source'
base64 -d <<'PXM_DENSITY_ARCHIVE' | tar -xz -C '{remote}/source'
{payload}
PXM_DENSITY_ARCHIVE
printf 'UPLOAD_OK bytes={len(archive.getvalue())}\\n'
'''
(task/'upload.sh').write_text(script,encoding='utf-8',newline='\n')
print('archive_bytes',len(archive.getvalue()))
