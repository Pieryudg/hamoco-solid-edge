#!/usr/bin/env python

import argparse
import os


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.description = 'Probe the local Solid Edge COM interface and list likely view command constants.'
    parser.add_argument('--launch', action='store_true', help='Launch Solid Edge if no running instance is found')
    parser.add_argument(
        '--terms',
        nargs='+',
        default=['view', 'fit', 'front', 'top', 'right', 'left', 'iso', 'zoom'],
        help='Case-insensitive constant-name terms to print',
    )
    args = parser.parse_args()

    if os.name != 'nt':
        raise SystemExit('Solid Edge COM probing is only available on Windows.')

    try:
        import win32com.client
    except ImportError as exc:
        raise SystemExit('pywin32 is required. Run scripts\\setup_windows.ps1 first.') from exc

    try:
        application = win32com.client.GetActiveObject('SolidEdge.Application')
        print('# Connected to running Solid Edge instance.')
    except Exception:
        if not args.launch:
            raise SystemExit('No running Solid Edge instance found. Start Solid Edge or rerun with --launch.')
        application = win32com.client.gencache.EnsureDispatch('SolidEdge.Application')
        print('# Launched Solid Edge through COM.')

    try:
        application.Visible = True
    except Exception:
        pass

    print(f'# Application: {getattr(application, "Name", "Solid Edge")}')
    try:
        print(f'# Version: {application.Version}')
    except Exception:
        pass
    try:
        print(f'# Active document: {application.ActiveDocument.Name}')
    except Exception:
        print('# Active document: none or unavailable')

    constants = win32com.client.constants
    terms = [term.lower() for term in args.terms]
    matches = []
    for name in dir(constants):
        lower_name = name.lower()
        if any(term in lower_name for term in terms):
            try:
                value = getattr(constants, name)
            except Exception:
                continue
            if isinstance(value, int):
                matches.append((name, value))

    if not matches:
        print('# No matching integer constants were available. Open Solid Edge once, then rerun this probe.')
        return

    print('# Candidate command constants:')
    for name, value in sorted(matches):
        print(f'{name}={value}')


if __name__ == '__main__':
    main()
