#!/usr/bin/env bash

SELFPLAY_PGN=PGN/selfplay.pgn

GAMES=$(find PGN_Selfplay -newer "${SELFPLAY_PGN}" -name "*.pgn")

if [ -z "$GAMES" ]; then
        echo "No new games found."
        exit 1
fi

TMP_FILE=$(mktemp -t selfplay.pgn)

trap "rm -f $TMP_FILE" EXIT

cat $GAMES "${SELFPLAY_PGN}" > $TMP_FILE

mv $TMP_FILE "${SELFPLAY_PGN}"

CNT=$(grep Result "${SELFPLAY_PGN}" | wc -l)

echo "${SELFPLAY_PGN} now has $CNT games."
