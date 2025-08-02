# License
#
# This software is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0
#
# Copyright (C) 2025 Roland Kohlhuber
#
# Note: The AI model used by this software (Qwen/Qwen3-Embedding-4B) retains its original license and is not subject to the AGPL license terms.
#
# For the complete license text, see: https://www.gnu.org/licenses/agpl-3.0.html

import grpc
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor
import random
import string
import sys

# --- IMPORTANT ---
# This script requires the generated protobuf files from 'embedding_qwen.proto'.
# Make sure 'embedding_qwen_pb2.py' and 'embedding_qwen_pb2_grpc.py' are in the same directory.
try:
    import embedding_qwen_pb2
    import embedding_qwen_pb2_grpc
except ImportError:
    print("Error: Could not import 'embedding_qwen_pb2.py' and 'embedding_qwen_pb2_grpc.py'.")
    print("Please run the protoc command to generate these files from 'embedding_qwen.proto'.")
    sys.exit(1)


class StressTest:
    """
    Handles sending gRPC requests to the Qwen3 server for the stress test.
    """
    def __init__(self, target_address="localhost:50051"):
        self.channel = grpc.insecure_channel(
            target_address,
            options=[
                ('grpc.max_send_message_length', -1),
                ('grpc.max_receive_message_length', -1),
                ('grpc.so_reuseport', 1),
                ('grpc.use_local_subchannel_pool', 1),
            ]
        )
        self.stub = embedding_qwen_pb2_grpc.EmbeddingServiceStub(self.channel)

    def generate_random_text(self):
        """Generates a short, random text string to simulate a unique user search query."""
        queries = [
            "how to build a fast API", "latest trends in machine learning",
            "what is vector search", "qwen3 embedding model",
            "compare python and rust", "benefits of async programming"
        ]
        base_text = random.choice(queries)
        random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        return f"{base_text} {random_suffix}"

    def test_single_search_embedding(self):
        """Sends a single embedding request to the gRPC server, simulating a search query."""
        text = self.generate_random_text()
        
        request = embedding_qwen_pb2.EmbedRequest(
            text=text,
            prompt_name="query" 
        )

        start_time = time.time()
        try:
            response = self.stub.Embed(request, timeout=15)
            return (time.time() - start_time, True)
        except grpc.RpcError as e:
            # You can uncomment the line below for detailed debugging of connection errors
            # print(f"gRPC Error: {e.code()} - {e.details()}")
            return (time.time() - start_time, False)

def run_test_level(tester: StressTest, num_users: int, duration: int):
    """
    Runs the stress test for a single concurrency level.
    """
    executor = ThreadPoolExecutor(max_workers=num_users)
    results = []
    
    stop_event = threading.Event()

    def worker_task():
        while not stop_event.is_set():
            results.append(tester.test_single_search_embedding())

    sys.stdout.write(f"  Testing with {num_users} users for {duration}s: [")
    sys.stdout.flush()

    [executor.submit(worker_task) for _ in range(num_users)]

    for _ in range(duration):
        time.sleep(1)
        sys.stdout.write("#")
        sys.stdout.flush()

    stop_event.set()
    executor.shutdown(wait=True)

    sys.stdout.write("] Done.\n")
    sys.stdout.flush()
    
    return results

def analyze_and_print_results(results, num_users, duration):
    """Analyzes the raw results and prints a formatted summary."""
    latencies = [r[0] for r in results if r[1]]
    error_count = len(results) - len(latencies)
    success_count = len(latencies)
    total_requests = len(results)

    header = f"| {'Concurrent Users':<18} | {'Total RPS':<12} | {'RPS/User':<10} | {'Avg Latency (ms)':<20} | {'P95 Latency (ms)':<20} | {'Errors':<10} |"
    separator = "-" * len(header)
    print(header)
    print(separator)

    if not latencies:
        print(f"| {num_users:<18} | {'N/A':<12} | {'N/A':<10} | {'N/A':<20} | {'N/A':<20} | {error_count:<10} |")
        print("\n🛑 Test stopped: No successful requests were made.")
        return False

    rps = success_count / duration
    rps_per_user = rps / num_users
    avg_latency = statistics.mean(latencies) * 1000
    p95_latency = statistics.quantiles(latencies, n=100)[94] * 1000
    success_rate = (success_count / total_requests) * 100 if total_requests > 0 else 0

    print(f"| {num_users:<18} | {rps:<12.2f} | {rps_per_user:<10.2f} | {avg_latency:<20.2f} | {p95_latency:<20.2f} | {error_count:<10} |")

    if success_rate < 99.5:
        print(f"\n🛑 Test stopped: Success rate dropped to {success_rate:.2f}% (< 99.5%).")
        return False
    if p95_latency > 1000:
        print(f"\n🛑 Test stopped: P95 latency exceeded 1000ms.")
        return False

    return True

def main():
    # --- Test Configuration ---
    GRPC_SERVER_ADDRESS = "localhost:50051"
    START_USERS = 25
    MAX_USERS = 500
    STEP = 25
    DURATION_PER_STEP = 15

    print("🚀 Starting Qwen3 gRPC Server Stress Test...")
    print(f"Targeting server: {GRPC_SERVER_ADDRESS}")
    
    tester = StressTest(GRPC_SERVER_ADDRESS)

    # --- MODIFIED: Health Check with More Patience ---
    print("\nChecking server connectivity...")
    connected = False
    max_retries = 8  # Increased from 5
    retry_delay = 3  # Increased from 2
    for i in range(max_retries):
        try:
            if tester.test_single_search_embedding()[1]:
                print("✅ gRPC server is responsive.")
                connected = True
                break
        except Exception:
            pass
        
        if i < max_retries - 1:
            print(f"  Connection attempt {i+1} failed. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    if not connected:
        print(f"❌ Could not connect to gRPC server at {GRPC_SERVER_ADDRESS} after {max_retries} attempts.")
        sys.exit(1)
    # --- End of Modification ---

    print("\n" + "=" * 110)

    for users in range(START_USERS, MAX_USERS + STEP, STEP):
        raw_results = run_test_level(tester, users, DURATION_PER_STEP)
        
        if not analyze_and_print_results(raw_results, users, DURATION_PER_STEP):
            break
        
        print("-" * 110)

    print("\n✅ Qwen3 stress test complete.")
    print("=" * 110)

if __name__ == "__main__":
    main()
