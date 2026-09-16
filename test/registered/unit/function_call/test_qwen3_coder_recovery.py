import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestQwen3CoderRecovery(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name=name,
                    parameters={
                        "type": "object",
                        "properties": {
                            "edits": {"type": "array"},
                            "obj": {"type": "object"},
                            "text": {"type": "string"},
                            "number": {"type": "integer"},
                        },
                    },
                ),
            )
            for name in ("edit", "second")
        ]

    @staticmethod
    def wrap(body, name="edit"):
        return f"<tool_call><function={name}>{body}</function></tool_call>"

    def decode_arguments(self, raw):
        def unique_object(pairs):
            result = dict(pairs)
            self.assertEqual(len(result), len(pairs), "Duplicate JSON argument keys")
            return result

        return json.loads(raw, object_pairs_hook=unique_object)

    def check_stream(self, chunks, expected):
        detector = Qwen3CoderDetector()
        calls = {}
        normal_text = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            for call in result.calls:
                item = calls.setdefault(call.tool_index, {"name": None, "raw": ""})
                if call.name is not None:
                    item["name"] = call.name
                item["raw"] += call.parameters
        self.assertEqual(normal_text, "")
        self.assertEqual(list(calls), list(range(len(expected))))
        self.assertEqual(
            [
                (call["name"], self.decode_arguments(call["raw"]))
                for call in calls.values()
            ],
            expected,
        )

    def check_modes(self, text, expected):
        with self.subTest(mode="nonstreaming"):
            result = Qwen3CoderDetector().detect_and_parse(text, self.tools)
            self.assertEqual(result.normal_text, "")
            self.assertEqual(
                [
                    (call.name, self.decode_arguments(call.parameters))
                    for call in result.calls
                ],
                expected,
            )
        with self.subTest(mode="streaming", chunks="full"):
            self.check_stream([text], expected)
        with self.subTest(mode="streaming", chunks="characters"):
            self.check_stream(text, expected)
        for split in range(1, len(text)):
            with self.subTest(mode="streaming", split=split):
                self.check_stream([text[:split], text[split:]], expected)

    def test_repeated_array_openings_between_elements(self):
        text = self.wrap(
            '<parameter=edits>\n[{"oldText":"a","newText":"b"},\n'
            '<parameter=edits>{"oldText":"c","newText":"d"},\n'
            '<parameter=edits>{"oldText":"e","newText":"f"}]\n</parameter>'
        )
        self.check_modes(
            text,
            [
                (
                    "edit",
                    {
                        "edits": [
                            {"oldText": "a", "newText": "b"},
                            {"oldText": "c", "newText": "d"},
                            {"oldText": "e", "newText": "f"},
                        ]
                    },
                )
            ],
        )

    def test_repeated_object_openings_between_members(self):
        text = self.wrap(
            '<parameter=obj>{"a":1,<parameter=obj>"b":2,'
            '<parameter=obj>"c":3}</parameter>'
        )
        self.check_modes(text, [("edit", {"obj": {"a": 1, "b": 2, "c": 3}})])

    def test_quoted_parameter_markers_and_escapes(self):
        values = [
            "literal <parameter=edits> marker",
            'escaped " quote <parameter=edits> marker',
            'backslash and quote \\" <parameter=edits> marker',
            "backslash \\ <parameter=text> other parameter",
            "trailing backslash \\",
        ]
        for value in values:
            for repeat in (False, True):
                with self.subTest(value=value, repeat=repeat):
                    marker = "<parameter=edits>" if repeat else ""
                    first = {"newText": value}
                    second = {"newText": "next"}
                    text = self.wrap(
                        f"<parameter=edits>[{json.dumps(first)},"
                        f"{marker}{json.dumps(second)}]</parameter>"
                    )
                    self.check_modes(text, [("edit", {"edits": [first, second]})])

    def test_single_quoted_python_literal_preserves_markers(self):
        text = self.wrap(
            r"""<parameter=edits>[{'newText': 'it\'s "quoted" \\ <parameter=edits> literal'},"""
            "<parameter=edits>{'newText': 'next'}]</parameter>"
        )
        self.check_modes(
            text,
            [
                (
                    "edit",
                    {
                        "edits": [
                            {"newText": 'it\'s "quoted" \\ <parameter=edits> literal'},
                            {"newText": "next"},
                        ]
                    },
                )
            ],
        )

    def test_quoted_object_keys_and_values(self):
        first = {"<parameter=obj>": 'quoted " <parameter=obj> value'}
        second = {"other": "<parameter=text>"}
        text = self.wrap(
            f"<parameter=obj>{json.dumps(first)[:-1]},"
            f"<parameter=obj>{json.dumps(second)[1:]}</parameter>"
        )
        self.check_modes(text, [("edit", {"obj": {**first, **second}})])

    def test_triple_quoted_python_literal_preserves_markers(self):
        for quote in ('"', "'"):
            with self.subTest(quote=quote):
                delimiter = quote * 3
                value = f"literal {quote} <parameter=edits> marker"
                text = self.wrap(
                    f"<parameter=edits>[{delimiter}{value}{delimiter}]</parameter>"
                )
                self.check_modes(text, [("edit", {"edits": [value]})])

    def test_plain_string_preserves_same_name_opening(self):
        for value in (
            "literal <parameter=text> marker",
            "first<parameter=text>second<parameter=text>third",
        ):
            with self.subTest(value=value):
                text = self.wrap(f"<parameter=text>{value}</parameter>")
                self.check_modes(text, [("edit", {"text": value})])

    def test_missing_close_before_different_parameter(self):
        text = self.wrap(
            "<parameter=edits>[1,2]<parameter=text>hello<parameter=number>7"
        )
        self.check_modes(
            text, [("edit", {"edits": [1, 2], "text": "hello", "number": 7})]
        )

    def test_unterminated_quote_preserves_next_parameter(self):
        text = self.wrap('<parameter=edits>["broken<parameter=text>hello</parameter>')
        self.check_modes(text, [("edit", {"edits": '["broken', "text": "hello"})])

    def test_missing_parameter_close_before_next_function(self):
        for shared_tool_call in (False, True):
            for second_close in ("", "</parameter>"):
                with self.subTest(
                    shared_tool_call=shared_tool_call, second_close=second_close
                ):
                    first = "<function=edit><parameter=edits>[1]</function>"
                    second = f"<function=second><parameter=edits>[2]{second_close}</function>"
                    separator = "" if shared_tool_call else "</tool_call><tool_call>"
                    text = f"<tool_call>{first}{separator}{second}</tool_call>"
                    self.check_modes(
                        text, [("edit", {"edits": [1]}), ("second", {"edits": [2]})]
                    )

    def test_closed_duplicate_keeps_first_value_in_both_modes(self):
        for name, first, second, expected in (
            ("edits", "[1]", "[2]", [1]),
            ("obj", '{"a":1}', '{"a":2}', {"a": 1}),
            ("text", "first", "second", "first"),
        ):
            with self.subTest(parameter=name):
                text = self.wrap(
                    f"<parameter={name}>{first}</parameter>"
                    f"<parameter={name}>{second}</parameter>"
                    f"<parameter={name}>{second}</parameter>"
                    "<parameter=number>7</parameter>"
                )
                self.check_modes(text, [("edit", {name: expected, "number": 7})])

    def test_duplicate_tracking_resets_between_functions(self):
        text = self.wrap(
            "<parameter=edits>[1]</parameter><parameter=edits>[9]</parameter>"
        ) + self.wrap(
            "<parameter=edits>[2]</parameter><parameter=edits>[8]</parameter>", "second"
        )
        self.check_modes(text, [("edit", {"edits": [1]}), ("second", {"edits": [2]})])

    def test_large_repeated_array(self):
        expected = list(range(2000))
        value = "[" + ",<parameter=edits>".join(map(str, expected)) + "]"
        text = self.wrap(f"<parameter=edits>{value}</parameter>")
        with self.subTest(mode="nonstreaming"):
            result = Qwen3CoderDetector().detect_and_parse(text, self.tools)
            self.assertEqual(len(result.calls), 1)
            self.assertEqual(
                self.decode_arguments(result.calls[0].parameters), {"edits": expected}
            )
        for chunk_size in (127, len(text)):
            with self.subTest(chunk_size=chunk_size):
                self.check_stream(
                    (
                        text[offset : offset + chunk_size]
                        for offset in range(0, len(text), chunk_size)
                    ),
                    [("edit", {"edits": expected})],
                )


if __name__ == "__main__":
    unittest.main()
