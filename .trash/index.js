import { CommitClassifier } from '../dist/index.min.mjs';

const classifier = new CommitClassifier(24, 0.1, 0.7);

// Dataset yang lebih variatif biar NN punya pola
const data = [
  // ==================== FEATURES ====================
  { text: 'feat: Add Doovy.ts machine learning engine', category: 'FEATURES'  },
  { text: 'feat: Add shellExec.ts for native command execution', category: 'FEATURES'  },
  { text: 'feat: Add security checker to action.yml workflow', category: 'FEATURES'  },
  { text: 'feat: Add copyright header verification system', category: 'FEATURES'  },
  { text: 'feat: Add detect trigger mechanism for repository events', category: 'FEATURES'  },
  { text: 'feat: Add calculate hash logic for file integrity', category: 'FEATURES'  },
  { text: 'feat: Add push-engine.ts core module development', category: 'FEATURES'  },
  { text: 'feat: Add index.ts export routing infrastructure', category: 'FEATURES'  },
  { text: 'feat: Added word order understanding feature with ngrams', category: 'FEATURES'  },
  { text: 'feat: Add on-the-fly compile runtime support', category: 'FEATURES'  },
  { text: 'implement lightvm core bytecode executor architecture', category: 'FEATURES'  },
  { text: 'feat: support multi-tenant database connection routing', category: 'FEATURES'  },
  { text: 'feat: implement oauth2 login flow with google provider', category: 'FEATURES'  },
  { text: 'feat: add support for webauthn biometric authentication', category: 'FEATURES'  },
  { text: 'feat: create dynamic dashboard widgets layout renderer', category: 'FEATURES'  },
  { text: 'feat: implement automatic database migration runner on startup', category: 'FEATURES'  },
  { text: 'feat: add native compression middleware support for response streams', category: 'FEATURES'  },
  { text: 'feat: introduce realtime notification engine using websocket server', category: 'FEATURES'  },
  { text: 'feat: implement rate limiting controller via redis memory cache', category: 'FEATURES'  },
  { text: 'feat(auth): support session revocation token list validation', category: 'FEATURES'  },
  { text: 'feat(api): expose query parameter metrics for performance analysis', category: 'FEATURES'  },
  { text: 'feat: add dark mode theme palette provider context', category: 'FEATURES'  },
  { text: 'feat: support markdown parsing engine inside rich text editor component', category: 'FEATURES'  },
  { text: 'feat: implement file upload processing stream chunks directly to s3 storage', category: 'FEATURES'  },
  { text: 'feat: introduce high performance bulk insertion framework for telemetry logs', category: 'FEATURES'  },
  { text: 'feat: support custom syntax configuration rules file loading', category: 'FEATURES'  },
  { text: 'feat: add automated email notification layout builder tool', category: 'FEATURES'  },
  { text: 'feat: implement hardware acceleration flags validation utility', category: 'FEATURES'  },
  { text: 'feat: add background worker queue state monitor module', category: 'FEATURES'  },
  { text: 'feat: implement incremental build mode options for fast development execution', category: 'FEATURES'  },

  // ==================== BUG_FIXES ====================
  { text: 'fix: Fix not exist tag error inside git engine', category: 'BUG_FIXES'  },
  { text: 'fix: Fix shellExec error msg not showing on output terminal', category: 'BUG_FIXES'  },
  { text: 'fix: Fix tag commit SHA error calculation mismatch', category: 'BUG_FIXES'  },
  { text: 'fix: Fix sync issues between remote and local pipeline', category: 'BUG_FIXES'  },
  { text: 'fix: Fix security checker boundary conditions vulnerabilities', category: 'BUG_FIXES'  },
  { text: 'fix: Fix cache logic storage path directory mismatch', category: 'BUG_FIXES'  },
  { text: 'fix: Fix install-compile.sh execution permission crash', category: 'BUG_FIXES'  },
  { text: 'fix: Fix sed-engine.sh syntax substitution bug', category: 'BUG_FIXES'  },
  { text: 'fix: comment respons string formatting parsing leak', category: 'BUG_FIXES'  },
  { text: 'fix: Fixed version writing inside automated text writer', category: 'BUG_FIXES'  },
  { text: 'fix: Fix NEW_VERSION bug variable tracking state override', category: 'BUG_FIXES'  },
  { text: 'fix: resolve unhandled promise rejection during connection timeout failure', category: 'BUG_FIXES'  },
  { text: 'fix: patch memory leak in event emitter listeners lookup maps', category: 'BUG_FIXES'  },
  { text: 'fix: prevent null pointer reference crash inside semantic syntax validator', category: 'BUG_FIXES'  },
  { text: 'fix: resolve deadlock blocking state in database transaction pool management', category: 'BUG_FIXES'  },
  { text: 'fix: fix broken responsive layout styling rules on mobile chrome browser view', category: 'BUG_FIXES'  },
  { text: 'fix: hotfix to filter out invalid authorization credentials tokens block', category: 'BUG_FIXES'  },
  { text: 'fix: resolve infinite loop execution during nested array serialization processes', category: 'BUG_FIXES'  },
  { text: 'fix: fix type mapping collision failure inside compiler configuration options', category: 'BUG_FIXES'  },
  { text: 'fix: prevent unexpected stack overflow boundary errors during deep recursion paths', category: 'BUG_FIXES'  },
  { text: 'fix(core): patch vulnerability validation schema checks mechanisms', category: 'BUG_FIXES'  },
  { text: 'fix(ui): correct modal dialog transition backdrop z-index placement stack', category: 'BUG_FIXES'  },
  { text: 'fix: escape dangerous characters to mitigate cross site scripting injection vectors', category: 'BUG_FIXES'  },
  { text: 'fix: catch file descriptor leak anomalies across active socket network streams', category: 'BUG_FIXES'  },
  { text: 'fix: restore corrupted session parameters during unexpected disconnect cycles', category: 'BUG_FIXES'  },
  { text: 'fix: fix environmental variable fallback definitions overriding production states', category: 'BUG_FIXES'  },
  { text: 'fix: resolve thread synchronization racing anomalies inside logging loop systems', category: 'BUG_FIXES'  },
  { text: 'fix: correct bad offset calculation formula reading binary blob allocations', category: 'BUG_FIXES'  },
  { text: 'fix: prevent cross origin cors blocker errors on dynamic file distribution points', category: 'BUG_FIXES'  },
  { text: 'fix: patch broken date format representation across alternative server locales', category: 'BUG_FIXES'  },

  // ==================== MAINTENANCE ====================
  { text: 'docs: Add CHANGELOG.md release documentation', category: 'MAINTENANCE'  },
  { text: 'chore: Add release.yml github action automatic workflow', category: 'MAINTENANCE'  },
  { text: 'chore: Add scripts directories for asset processing automation', category: 'MAINTENANCE'  },
  { text: 'update Doovy AI parameters and model internal properties', category: 'MAINTENANCE'  },
  { text: 'refactor: Clean up code complexity inside semantic optimizer', category: 'MAINTENANCE'  },
  { text: 'feat: Update Doovy dataset vocabulary limits for tuning', category: 'MAINTENANCE'  },
  { text: 'chore: Setup prettier format rules configuration styling', category: 'MAINTENANCE'  },
  { text: 'chore: Update types declaration definitions interfaces', category: 'MAINTENANCE'  },
  { text: 'refactor: Tidy up the log step code structure sequence', category: 'MAINTENANCE'  },
  { text: 'chore: Tidying up the code and clearing unused imports', category: 'MAINTENANCE'  },
  { text: 'feat: Update push-engine log msg format styling lines', category: 'MAINTENANCE'  },
  { text: 'feat: Update action.yml metadata properties parameters', category: 'MAINTENANCE'  },
  { text: 'chore: Update package.json scripts run and target versions', category: 'MAINTENANCE'  },
  { text: 'chore: bump dependencies version parameters to comply with upstream patches', category: 'MAINTENANCE'  },
  { text: 'docs: update markdown documentation setup guide detailing deployment tutorials', category: 'MAINTENANCE'  },
  { text: 'refactor: decouple network routing adapters architecture layer completely', category: 'MAINTENANCE'  },
  { text: 'chore: configure strict eslint tracking parameters across root source folders', category: 'MAINTENANCE'  },
  { text: 'perf: optimize internal payload serialization performance speeds throughput', category: 'MAINTENANCE'  },
  { text: 'test: expand unit test assertions coverage limits validating server routines', category: 'MAINTENANCE'  }, 
  { text: 'chore: clean up obsolete temporary artifact tracks inside gitignore profiles', category: 'MAINTENANCE'  },
  { text: 'refactor: simplify deeply nested conditional evaluation statements logic structures', category: 'MAINTENANCE'  },
  { text: 'docs: fix grammatical typo mistakes across API reference documentation pages', category: 'MAINTENANCE'  },
  { text: 'chore(ci): split test running tasks across parallel cloud workers setups', category: 'MAINTENANCE'  },
  { text: 'style: normalize code blocks indentation rules and trailing space behaviors', category: 'MAINTENANCE'  },
  { text: 'perf: rewrite inner math computation loop using lookup tables cache arrays', category: 'MAINTENANCE'  },
  { text: 'chore: migrate old configuration manifests structures to updated schema models', category: 'MAINTENANCE'  },
  { text: 'refactor: rename abstract factory base interfaces eliminating token ambiguity', category: 'MAINTENANCE'  },
  { text: 'chore: audit corporate license header stamps compliance across package trees', category: 'MAINTENANCE'  },
  { text: 'test: add integration benchmark testing coverage checking worker pooling strain', category: 'MAINTENANCE'  },
  { text: 'chore: prune deprecated helper wrappers routines legacy methods files', category: 'MAINTENANCE'  },
  
  // ==================== TAMBAHAN: GENERAL PURPOSE ====================
  // FEATURES
  { text: 'feat: Add user profile avatar upload functionality', category: 'FEATURES' },
  { text: 'feat: Implement password reset via email link', category: 'FEATURES' },
  { text: 'feat: Add social media login integration', category: 'FEATURES' },
  { text: 'feat: Enable multi-language support for UI', category: 'FEATURES' },
  { text: 'feat: Add PDF export capability for reports', category: 'FEATURES' },
  { text: 'feat: Implement infinite scroll for product list', category: 'FEATURES' },
  { text: 'feat: Add search bar with autocomplete suggestions', category: 'FEATURES' },
  { text: 'feat: Add dark mode toggle in settings', category: 'FEATURES' },
  
  // BUG_FIXES
  { text: 'fix: Resolve login timeout for slower networks', category: 'BUG_FIXES' },
  { text: 'fix: Correct button alignment on smaller screens', category: 'BUG_FIXES' },
  { text: 'fix: Remove duplicate entries in user list', category: 'BUG_FIXES' },
  { text: 'fix: Handle empty API response gracefully', category: 'BUG_FIXES' },
  { text: 'fix: Fix incorrect timezone display in dashboard', category: 'BUG_FIXES' },
  { text: 'fix: Prevent form submission with invalid email format', category: 'BUG_FIXES' },
  { text: 'fix: Resolve memory leak when switching tabs', category: 'BUG_FIXES' },
  { text: 'fix: Ensure scroll position is reset on page change', category: 'BUG_FIXES' },
  
  // MAINTENANCE
  { text: 'chore: Update project dependencies to stable versions', category: 'MAINTENANCE' },
  { text: 'docs: Improve README with installation steps', category: 'MAINTENANCE' },
  { text: 'refactor: Move API constants to central config file', category: 'MAINTENANCE' },
  { text: 'test: Add unit tests for auth service', category: 'MAINTENANCE' },
  { text: 'chore: Clean up unused CSS classes', category: 'MAINTENANCE' },
  { text: 'style: Reformat code with prettier', category: 'MAINTENANCE' },
  { text: 'refactor: Rename state variables for better clarity', category: 'MAINTENANCE' },
  { text: 'docs: Update license information in root directory', category: 'MAINTENANCE' }
];

console.log("🚀 Starting Training...");
// Epoh dinaikin biar konvergen
classifier.train(data, 700);

// Test Klasifikasi
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

// Test Changelog
const changelog = classifier.generateChangelog([
  "feat: add dark mode",
  "fix: login button alignment",
  "feat: add dark mode", // Harusnya ke-filter sebagai duplikat
  "chore: clean temp files",
  "fix: resolve infinite loop in fetch",
  "fix subagia kena bug turunin feature chore monyet",
  "fix: fix fix fix",
  "finally make this thing stop exploding",
  "remove workaround because workaround for workaround no longer needed",
  "please work"
]);

console.log("\n📄 Changelog Preview:");
console.log(changelog);
