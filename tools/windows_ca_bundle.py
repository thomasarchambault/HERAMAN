"""Build a pip CA bundle from public roots trusted by Windows; never disable TLS."""
import os
import ssl
import sys
from pathlib import Path


def main():
    if os.name!='nt':
        raise RuntimeError('This certificate bootstrap is for Windows only.')
    from pip._vendor import certifi
    certificates = [Path(certifi.where()).read_text(encoding='ascii')]
    for certificate, encoding, trust in ssl.enum_certificates('ROOT'):
        if encoding=='x509_asn' and (trust is True or '1.3.6.1.5.5.7.3.1' in trust):
            certificates.append(ssl.DER_cert_to_PEM_cert(certificate))
    target=Path(sys.argv[1]).resolve()
    target.write_text('\n'.join(certificates),encoding='ascii')
    print(target)


if __name__=='__main__':
    main()
