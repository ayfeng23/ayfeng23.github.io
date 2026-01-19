#!/bin/bash
# update-dates.sh - Updates last edited dates in HTML files based on git history

for file in *.html; do
  # Get the last commit date for this file
  date=$(git log -1 --format=%cd --date=format:'%B %d, %Y' -- "$file")

  # Skip if no git history for this file
  if [ -z "$date" ]; then
    continue
  fi

  # Check if there's already a date footer
  if grep -q '<div class="last-edited">' "$file"; then
    # Update existing date
    sed -i '' "s|<div class=\"last-edited\">Last edited: .*</div>|<div class=\"last-edited\">Last edited: $date</div>|" "$file"
  else
    # Add date before closing </body> tag
    sed -i '' "s|</body>|  <div class=\"last-edited\">Last edited: $date</div>\n</body>|" "$file"
  fi
done

echo "Updated last edited dates for all HTML files"
