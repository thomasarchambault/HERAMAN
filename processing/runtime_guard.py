# Rôle : Empêcher l’exécution des workers scientifiques avec un interpréteur non 64 bits.

import struct


def require_64bit() -> None:
    bits = struct.calcsize("P") * 8
    if bits != 64:
        raise RuntimeError(
            f"Ce traitement doit être lancé avec Python 64 bits; "
            f"l'interpréteur actuel est {bits} bits."
        )

