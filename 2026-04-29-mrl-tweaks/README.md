# MRL 2026 Shared Task Utils

Split multi-language cells into one language per row and add ISO 639-3 codes:

```bash
python dedup_entries.py -i contact_leads.csv -o contact_leads_dedup.csv
```

The output preserves all original columns, normalizes language naming variants (via `langcodes` + aliases), and adds:
1. `Language_Original` (exact raw text from the original `Language` cell)
2. `Language_ISO3`
