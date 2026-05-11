# Proposed Claude Skills Structure

```
llm-compressor/
├── .claude/
│   └── skills/
│       └── write-recipe.md
│
├── src/llmcompressor/
│   ├── modeling/
│   │   └── .claude/
│   │       └── skills/
│   │           ├── add-model.md
│   │           └── add-moe-calibration.md
│   │
│   └── modifiers/
│       └── .claude/
│           └── skills/
│               └── add-modifier.md
│
├── examples/
│   └── .claude/
│       └── skills/
│           └── write-example.md
│
└── tests/
    ├── .claude/
    │   └── skills/
    │       └── write-unit-test.md
    │
    ├── lmeval/
    │   └── .claude/
    │       └── skills/
    │           └── run-lmeval.md
    │
    ├── e2e/
    │   └── .claude/
    │       └── skills/
    │           └── run-e2e.md
    │
    └── examples/
        └── .claude/
            └── skills/
                └── verify-examples.md
```

## Priority

| Priority | Skill |
|---|---|
| 1 | `add-moe-calibration` |
| 2 | `run-lmeval` |
| 3 | `write-example` |
| 4 | `add-modifier` |
| 5 | `run-e2e` |
| 6 | `verify-examples` |
| 7 | `write-unit-test` |
| 8 | `add-model` |
