#!/bin/bash

INDEX_FILE="source/examples.md"
DOC_FOLDER="source/doc"

cat > "$INDEX_FILE" <<EOF
# Usage examples
EOF

# Loop through .md files and generate individual sections
find "$DOC_FOLDER" -type f -name '*.md' | sort | while read -r file; do
    title=$(grep -m1 '^# ' "$file" | sed 's/^# //' | sed 's/ *#*$//')
    rel_path=$(echo "$file" | sed 's|^.*source/||')  # Strip leading source/ folder
    echo -e "\n% $title\n" >> "$INDEX_FILE"
    echo ":::{toctree}" >> "$INDEX_FILE"
    echo ":maxdepth: 1" >> "$INDEX_FILE"
    echo "" >> "$INDEX_FILE"
    echo "$rel_path" >> "$INDEX_FILE"
    echo -e "\n:::\n" >> "$INDEX_FILE"
done
