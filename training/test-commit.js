import { CommitClassifier } from '../dist/index.min.mjs';
import fs from "fs";

const classifier = new CommitClassifier(64, 0.02, 0.75, 'swish');

classifier.loadModel('./training/models/doovy-source-001.json');

function main() {
  const tests = JSON.parse(fs.readFileSync("./training/tests.json", "utf8"));
  
  console.log("\n🧪 Hasil Testing:");
  tests.forEach(t => {
    console.log(`Input: "${t}" -> Hasil: ${classifier.classify(t)}`);
  });
  
  const changelog = classifier.generateChangelog(tests);
  
  console.log("\n📄 Changelog Preview:");
  console.log(changelog);
}

main();