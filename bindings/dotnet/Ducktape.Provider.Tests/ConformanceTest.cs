using Xunit;

namespace Ducktape.Tests;

public class ConformanceTest
{
    public static IEnumerable<object[]> WireFixtures()
    {
        return Fixtures.All().Select(path => new object[] { Path.GetRelativePath(Fixtures.Wire, path) });
    }

    [Theory]
    [MemberData(nameof(WireFixtures))]
    public void Replays(string relative)
    {
        var fixture = Fixtures.Load(Path.Combine(Fixtures.Wire, relative));
        var got = Fixtures.Replay(fixture);
        var expected = fixture["exchange"]!.AsArray();
        Assert.True(expected.Count == got.Count, $"length {got.Count} != {expected.Count}; last {got[^1].Step.ToJsonString()}");
        for (var i = 0; i < got.Count; i++)
        {
            Assert.Equal(Fixtures.Canonical(expected[i]!["step"]), Fixtures.Canonical(got[i].Step));
            Assert.Equal(Fixtures.Canonical(expected[i]!["input"]), Fixtures.Canonical(got[i].Input));
        }
    }
}
