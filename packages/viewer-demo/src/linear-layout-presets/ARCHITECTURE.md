The preset catalog is split by instruction because contributors usually know one instruction family well and should be able to edit only that slice.

Each file in this directory exports raw preset definitions for one instruction family:

- `mma.ts`
- `wgmma.ts`
- `ldmatrix.ts`
- `stmatrix.ts`
- `swizzle.ts`

Those files are intentionally data-heavy and light on control flow. The goal is to keep instruction knowledge declarative: names, signatures, comments, and basis rows live next to each other, and `linear-layout.ts` stays responsible for turning that data into UI presets.

`types.ts` defines the shared preset shapes. If a contributor wants to add a new instruction family, the expected workflow is:

1. Add one new file in this directory that exports `ComposeLayoutPresetDefinition[]`.
2. Keep instruction-specific naming/comments in that file.
3. Import and append that array in `../linear-layout.ts`.

This avoids one giant preset file, reduces merge conflicts, and makes it clear where instruction-specific review should happen.
