namespace Ducktape.Types;

/// <summary>Known values of AuthSource.preset; other strings are allowed.</summary>
public static class AuthSourcePreset
{
    public const string AnthropicProfile = "anthropic_profile";
    public const string ClaudeCode = "claude_code";
    public const string ClaudeCodeTokenEnv = "claude_code_token_env";
    public const string Codex = "codex";
    public const string ChatgptPlan = "chatgpt_plan";
}
