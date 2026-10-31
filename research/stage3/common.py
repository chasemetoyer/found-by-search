"""Stage3 constants; scientific Stage1/2 files remain immutable."""
from stage2.common import ROOT,PROJECT_ID,atomic_json,canonical_sha,parse_cpuinfo,cpu_class,sha256_file,utc_now
APP_NAME='evolved-intrinsic-rewards-stage3'
VOLUME_NAME=APP_NAME
VOLUME_VERSION=2
CONTROLLER_LOCK_NAME=APP_NAME+'-controller'
MAX_CONCURRENT_CALLS=32
FUNCTION_TIMEOUT_SECONDS=3600
STARTUP_TIMEOUT_SECONDS=300
CONTROLLER_TIMEOUT_SECONDS=43200
HARD_CAP_USD='15'
REVIEW_THRESHOLD_USD='13'
