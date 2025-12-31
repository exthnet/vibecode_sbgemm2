#!/bin/bash
# エージェントローカル起動スクリプト

set -e

# プロジェクトルートは環境変数から取得
if [ -z "$VIBECODE_ROOT" ]; then
    echo "❌ Error: VIBECODE_ROOT not set"
    exit 1
fi

# エージェントIDを引数から取得
AGENT_ID=$1
shift

# エージェントタイプを判定
determine_agent_type() {
    local agent_id=$1
    if [[ "$agent_id" =~ ^(PM|SE|PG|CD) ]]; then
        echo "polling"
    else
        echo "event-driven"
    fi
}

AGENT_TYPE=$(determine_agent_type "$AGENT_ID")
AGENT_DIR="$(pwd)"

echo "🔧 Setting up agent $AGENT_ID (type: $AGENT_TYPE)"

# Hooksを設定（VIBECODE_ENABLE_HOOKSがfalseでない限り有効）
if [ "${VIBECODE_ENABLE_HOOKS}" != "false" ]; then
    # CLI_HOOKS_MODEを取得（環境変数から、デフォルトはauto）
    CLI_HOOKS_MODE="auto"
    if [ -f "$VIBECODE_ROOT/hooks/setup_agent_hooks.sh" ]; then
        "$VIBECODE_ROOT/hooks/setup_agent_hooks.sh" "$AGENT_ID" "$AGENT_DIR" "$AGENT_TYPE" "$CLI_HOOKS_MODE"
    else
        echo "⚠️  Warning: setup_agent_hooks.sh not found"
    fi
fi

# working_dirをJSONLテーブルに記録
if command -v jq &> /dev/null; then
    TABLE_FILE="$VIBECODE_ROOT/Agent-shared/agent_and_pane_id_table.jsonl"
    if [ -f "$TABLE_FILE" ]; then
        echo "📝 Updating working_dir for $AGENT_ID"
        WORKING_DIR="${AGENT_DIR#$VIBECODE_ROOT/}"
        
        # 一時ファイルを使用して更新
        TEMP_FILE="$TABLE_FILE.tmp"
        while IFS= read -r line; do
            if [[ -z "$line" || "$line" =~ ^# ]]; then
                echo "$line"
            else
                updated_line=$(echo "$line" | jq -c --arg id "$AGENT_ID" --arg dir "$WORKING_DIR" '
                    if .agent_id == $id then
                        . + {working_dir: $dir, last_updated: (now | strftime("%Y-%m-%dT%H:%M:%SZ"))}
                    else
                        .
                    end
                ')
                echo "$updated_line"
            fi
        done < "$TABLE_FILE" > "$TEMP_FILE"
        mv "$TEMP_FILE" "$TABLE_FILE"
        echo "✅ Updated working_dir to: $WORKING_DIR"
    fi
fi

# PM/SE/PGエージェントの場合、MCP（Desktop Commander）を設定
if [[ "$AGENT_ID" =~ ^(PM|SE|PG) ]]; then
    echo "🔧 Setting up MCP for $AGENT_ID agent"
    claude mcp add desktop-commander -- npx -y @wonderwhy-er/desktop-commander
fi

# テレメトリ設定に基づいてClaude起動
if [ "${VIBECODE_ENABLE_TELEMETRY}" = "false" ]; then
    echo "📊 Telemetry disabled - starting agent without telemetry"
    exec env DISABLE_AUTOUPDATER=1 claude --dangerously-skip-permissions "$@"
else
    echo "📊 Telemetry enabled - starting agent with telemetry"
    exec "$VIBECODE_ROOT/telemetry/launch_claude_with_env.sh" "$AGENT_ID" "$@"
fi
