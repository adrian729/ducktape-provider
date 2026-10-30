namespace Ducktape.Types;

/// <summary>Known values of AuthSource.grant; other strings are allowed.</summary>
public static class AuthSourceGrant
{
    public const string RefreshToken = "refresh_token";
    public const string JwtBearer = "jwt_bearer";
    public const string ClientCredentials = "client_credentials";
    public const string StsWebIdentity = "sts_web_identity";
    public const string GcpMetadata = "gcp_metadata";
    public const string AwsImds = "aws_imds";
    public const string AwsContainer = "aws_container";
}
