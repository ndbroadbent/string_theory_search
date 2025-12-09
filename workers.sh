#!/bin/bash
# Manage string theory search workers
# Usage: ./workers.sh [start|stop|status|restart] [count]

ACTION=${1:-status}
COUNT=${2:-2}
MAX_WORKERS=10

case "$ACTION" in
  start)
    echo "Starting $COUNT worker(s)..."
    for i in $(seq 1 $COUNT); do
      systemctl start string-theory-search@$i
    done
    sleep 2
    $0 status
    ;;
  stop)
    echo "Stopping all workers..."
    for i in $(seq 1 $MAX_WORKERS); do
      systemctl stop string-theory-search@$i 2>/dev/null
    done
    sleep 1
    $0 status
    ;;
  restart)
    $0 stop
    sleep 1
    $0 start $COUNT
    ;;
  status)
    echo "Worker Status:"
    echo "=============="
    RUNNING=0
    for i in $(seq 1 $MAX_WORKERS); do
      STATUS=$(systemctl is-active string-theory-search@$i 2>/dev/null)
      if [ "$STATUS" = "active" ]; then
        echo "  Worker $i: RUNNING"
        RUNNING=$((RUNNING + 1))
      fi
    done
    if [ $RUNNING -eq 0 ]; then
      echo "  No workers running"
    else
      echo ""
      echo "Total: $RUNNING worker(s) running"
    fi
    ;;
  logs)
    WORKER=${2:-1}
    journalctl -u string-theory-search@$WORKER -f
    ;;
  *)
    echo "Usage: $0 [start|stop|status|restart|logs] [count|worker_id]"
    echo ""
    echo "Commands:"
    echo "  start [n]    - Start n workers (default: 2)"
    echo "  stop         - Stop all workers"
    echo "  restart [n]  - Restart with n workers"
    echo "  status       - Show running workers"
    echo "  logs [n]     - Follow logs for worker n (default: 1)"
    ;;
esac
