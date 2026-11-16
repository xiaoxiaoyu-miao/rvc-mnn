#!/usr/bin/env python3
"""通过 uiautomator + input tap 控制 App（Termux 上 adb 不可用，走 root）。"""
import re
import subprocess
import sys
import time


def sh(cmd):
    return subprocess.run(['su', '-c', cmd], capture_output=True, text=True)


def dump():
    sh('uiautomator dump /data/local/tmp/ui.xml')
    return sh('cat /data/local/tmp/ui.xml').stdout


def find_nodes(xml, needle):
    """返回 [(text, cx, cy), ...]，按 text 子串匹配。"""
    out = []
    if not xml:
        return out
    for m in re.finditer(r'text="([^"]*)"[^>]*?bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"', xml):
        t = m.group(1)
        if needle in t:
            x1, y1, x2, y2 = map(int, m.groups()[1:])
            out.append((t.replace('&#10;', ' ')[:60], (x1 + x2) // 2, (y1 + y2) // 2))
    return out


def tap(x, y):
    sh(f'input tap {x} {y}')


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'list'

    if action == 'list':
        sh('am start -n com.rvc.mnn/.MainActivity')
        time.sleep(3)
        for n in find_nodes(dump(), sys.argv[2] if len(sys.argv) > 2 else ''):
            print(n)
        return

    if action == 'tap':
        needle = sys.argv[2]
        sh('am start -n com.rvc.mnn/.MainActivity')
        time.sleep(2.5)
        nodes = find_nodes(dump(), needle)
        if not nodes:
            print('NOT FOUND:', needle)
            return 1
        t, cx, cy = nodes[0]
        print(f'TAP "{t}" @ ({cx},{cy})')
        tap(cx, cy)
        return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
