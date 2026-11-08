"""Explicit identity-bound supervisor handoff; never starts or reconfigures ranks."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import time


def stamp(pid):
    p = Path('/proc') / str(pid)
    parts = (p/'stat').read_text().rsplit(')', 1)[1].split()
    if parts[0] == 'Z':
        raise ProcessLookupError(pid)
    return parts[19], (p/'cmdline').read_bytes().split(b'\0')


class AdoptedWorker:
    def __init__(self, pid, root, receipt):
        self.pid, self.root, self.returncode = pid, Path(root), None
        self.start, command = stamp(pid)
        expected = receipt['worker']
        if int(expected['pid']) != pid or expected['start_ticks'] != self.start:
            raise ValueError('Adopted worker identity differs')
        if b'train.monitored' not in command or str(self.root).encode() not in command:
            raise ValueError('Adopted command/run differs')
        self.attempt = command[command.index(b'--attempt') + 1].decode()
        children = [int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
        if set(children) != {int(x) for x in receipt['ranks']} or len(children) != 8:
            raise ValueError('Adoption requires the original eight ranks')
        self.ranks = receipt['ranks']
        for rank_pid, start in self.ranks.items():
            actual, cmd = stamp(int(rank_pid))
            if actual != start or b'train.monitored' not in cmd or str(self.root).encode() not in cmd:
                raise ValueError('Rank identity differs')

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        try:
            start, _ = stamp(self.pid)
            if start != self.start:
                raise RuntimeError('Adopted PID reused')
            return None
        except (FileNotFoundError, ProcessLookupError):
            # An orphan's kernel exit code may be reaped by init. Require each
            # producer rank's successful completion and final seal instead.
            complete = (self.root/'training/ckpt_0057000/.complete').is_file()
            for rank in range(8):
                p = self.root/'ledgers'/f'{self.attempt}-rank{rank}.json'
                row = json.loads(p.read_text()) if p.is_file() else {}
                complete = complete and row.get('success') is True and row.get('checkpoint_step') == 57000
            self.returncode = 0 if complete else 1
            return self.returncode

    def wait(self, timeout=None):
        end = time.monotonic() + timeout if timeout is not None else float('inf')
        while self.poll() is None:
            if time.monotonic() >= end:
                raise subprocess.TimeoutExpired(str(self.pid), timeout)
            time.sleep(.1)
        return self.returncode

    def stop_owned(self):
        """Preserve PID/start-time boundaries even if torchrun has already died."""
        expected={str(self.pid):self.start,**self.ranks}
        def remaining():
            found=[]
            for pid,start in expected.items():
                try:
                    actual,cmd=stamp(int(pid))
                    if actual==start and b'train.monitored' in cmd and str(self.root).encode() in cmd:
                        found.append(int(pid))
                except (FileNotFoundError,ProcessLookupError):pass
            return found
        for pid in remaining():
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        end=time.monotonic()+5
        while remaining() and time.monotonic()<end:time.sleep(.1)
        for pid in remaining():
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass


def gpu_lock(worker):
    from common.data_contracts import resolve
    lock = resolve('runs/.gpu_benchmark.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if {int(x.strip()) for x in occupied.splitlines() if x.strip()} != {int(p) for p in worker.ranks}:
            raise ValueError('GPU ownership differs from adopted ranks')
        if worker.poll() is not None:
            raise ValueError('Worker ended during handoff')
        return lock
    except BaseException:
        lock.close()
        raise
