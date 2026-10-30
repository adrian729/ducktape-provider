package io.github.adrian729.ducktape;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.fasterxml.jackson.databind.JsonNode;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.TestFactory;

class ConformanceTest {
    @TestFactory
    Stream<DynamicTest> wireFixtures() {
        Core core = Core.bundled();
        return Fixtures.all().stream().map(path -> DynamicTest.dynamicTest(path.getParent().getFileName() + "/" + path.getFileName(), () -> check(core, path)));
    }

    private static void check(Core core, Path path) {
        JsonNode fixture = Fixtures.load(path);
        List<JsonNode[]> got = Fixtures.replay(core, fixture);
        JsonNode expected = fixture.get("exchange");
        assertEquals(expected.size(), got.size(), () -> "length; last " + got.get(got.size() - 1)[0]);
        for (int i = 0; i < got.size(); i++) {
            assertEquals(Fixtures.canonical(expected.get(i).get("step")), Fixtures.canonical(got.get(i)[0]), "step " + i);
            assertEquals(Fixtures.canonical(expected.get(i).get("input")), Fixtures.canonical(got.get(i)[1]), "input " + i);
        }
    }
}
