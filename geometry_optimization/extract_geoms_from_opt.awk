#!/usr/bin/awk -f

/Z \(Atomic Numbers\).*Geom/ {
    capture = 1
    atom_count = 0
    next
}


/- Coordinate -/ {
    capture = 0
    if (atom_count > 0) {
        print atom_count "\nStep " ++step
        for (i = 1; i <= atom_count; i++) {
            print lines[i]
        }
        atom_count = 0
    }
    next
}

capture && NF >= 5 && $1 ~ /^[0-9.]/ {
    atom_count++
    atomZ = $1
    x = $(3) * 0.529177
    y = $(4) * 0.529177
    z = $(5) * 0.529177

    # Translate atomic number to element symbol
    if (atomZ == 1.0)       sym = "H"
    else if (atomZ == 6.0)  sym = "C"
    else if (atomZ == 7.0)  sym = "N"
    else if (atomZ == 8.0)  sym = "O"
    else                    sym = "X"  # Unknown

    lines[atom_count] = sym "  " x "  " y "  " z
}
