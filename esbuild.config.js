import esbuild from 'esbuild';
const builds = [
  {
    entryPoints: ['./src/index.ts'],
    format: 'esm',
    outfile: './dist/index.min.mjs',
  },
  {
    entryPoints: ['./src/index.ts'],
    format: 'cjs',
    outfile: './dist/index.min.cjs',
  },
];
for (const config of builds) {
  esbuild
    .build({
      bundle: true,
      minify: true,
      sourcemap: true,
      platform: 'node',
      ...config,
      external: ['chalk', '@lightvm/core-*'],
    })
    .catch(() => process.exit(1));
}
