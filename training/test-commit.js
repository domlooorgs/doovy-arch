import { CommitClassifier } from '../dist/index.min.mjs';

const classifier = new CommitClassifier(64, 0.02, 0.75, 'swish');

classifier.loadModel('./training/models/doovy-source-001.json');

function main() {
  const tests = [
    "fix: button alignment issue",
    "feat: add export to pdf",
    "chore: bump version",
    "fix subagia kena bug turunin feature chore monyet"
  ];
  
  console.log("\n🧪 Hasil Testing:");
  tests.forEach(t => {
    console.log(`Input: "${t}" -> Hasil: ${classifier.classify(t)}`);
  });
  
  const changelog = classifier.generateChangelog([
    "feat: add dark mode",
    "fix: login button alignment",
    "feat: add dark mode",
    "chore: clean temp files",
    "fix: resolve infinite loop in fetch",
    "fix: fix fix fix",
    "finally make this thing stop exploding",
    "remove workaround because workaround for workaround no longer needed",
    "please work",
    "fix subagia kena bug turunin feature chore monyet",
    "capek gua ngentot!",
    "fix feat bug chore refactor"
  ]);
  
  console.log("\n📄 Changelog Preview:");
  console.log(changelog);
}

main();