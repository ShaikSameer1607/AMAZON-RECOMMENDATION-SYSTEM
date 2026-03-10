#!/bin/bash
# Pipeline Progress Monitor

echo "=========================================="
echo "Amazon Recommendation System - Progress"
echo "=========================================="
echo ""

# Check if any Python/Spark process is running
echo "Active Processes:"
ps aux | grep -E "(python.*spark|spark.*python)" | grep -v grep | awk '{print "  PID:", $2, "| CPU:", $3"%", "| Memory:", $4"%"}'
echo ""

# Check YARN applications
echo "YARN Applications:"
yarn application -list 2>/dev/null | grep -E "(RUNNING|ACCEPTED)" | head -5 || echo "  No active YARN applications"
echo ""

# Check HDFS health
echo "HDFS Status:"
hdfs dfsadmin -report 2>/dev/null | grep -E "(Configured Capacity|DFS Remaining|Under replicated blocks)" | head -3 || echo "  Cannot reach HDFS"
echo ""

# Check disk space
echo "Local Disk Space:"
df -h /home/sameer | tail -1 | awk '{print "  Used:", $3, "| Available:", $4}'
echo ""

# Show recent log entries if pipeline.log exists
if [ -f "pipeline.log" ]; then
    echo "Recent Logs (last 10 lines):"
   tail -10 pipeline.log
else
    echo "No pipeline.log found"
fi
echo ""

echo "=========================================="
echo "Tips:"
echo "  - Full pipeline takes 30-40 minutes for 6.7GB"
echo "  - Check Spark UI at http://localhost:4040 during execution"
echo "  - For faster testing, use reduced dataset"
echo "=========================================="
