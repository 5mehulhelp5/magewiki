from pathlib import Path

from app.scanner.php_classes import (
    _string_literal_value,
    catalog_module_classes,
    entry_from_dict,
    entry_to_dict,
    parse_php_file,
    parse_php_source,
)
from app.scanner.php_parser import find_all, parse_php


def test_catalog_module_classes(demo_module_path: Path):
    classes = catalog_module_classes(demo_module_path)
    by_name = {c.class_name: c for c in classes}

    assert "SampleCheckoutPlugin" in by_name
    assert "SampleOrderObserver" in by_name

    plugin = by_name["SampleCheckoutPlugin"]
    assert plugin.namespace == "Vendor\\Module\\Plugin"
    assert plugin.fqcn == "Vendor\\Module\\Plugin\\SampleCheckoutPlugin"
    assert plugin.kind == "class"
    assert plugin.file_path == "Plugin/SampleCheckoutPlugin.php"
    assert {m.name for m in plugin.methods} == {"beforeSubmitAll"}

    observer = by_name["SampleOrderObserver"]
    assert observer.namespace == "Vendor\\Module\\Observer"
    assert observer.implements == ["ObserverInterface"]
    assert {m.name for m in observer.methods} == {"execute"}


def test_parse_php_file_uses_each_class_own_namespace_block(tmp_path: Path):
    # Two braced namespace blocks in one file — each class must get its OWN
    # enclosing namespace, not the file's first one.
    braced = tmp_path / "Braced.php"
    braced.write_text(
        "<?php\n"
        "namespace A {\n"
        "    class Foo {}\n"
        "}\n"
        "namespace B {\n"
        "    class Bar {}\n"
        "}\n"
    )
    entries = {e.class_name: e for e in parse_php_file(braced, "Braced.php")}
    assert entries["Foo"].fqcn == "A\\Foo"
    assert entries["Bar"].fqcn == "B\\Bar"

    # Semicolon-style: a namespace applies to every following declaration up
    # to the next namespace statement, not to the whole file.
    semicolon = tmp_path / "Semicolon.php"
    semicolon.write_text(
        "<?php\n"
        "namespace A;\n"
        "class Foo {}\n"
        "namespace B;\n"
        "class Bar {}\n"
    )
    entries = {e.class_name: e for e in parse_php_file(semicolon, "Semicolon.php")}
    assert entries["Foo"].fqcn == "A\\Foo"
    assert entries["Bar"].fqcn == "B\\Bar"


def test_catalog_module_classes_skips_a_file_that_vanishes_before_it_can_be_read(tmp_path: Path, monkeypatch):
    # A TOCTOU race (the file existed when discovered but is gone by the time it's
    # read) is the realistic way this happens in production; simulated here by
    # having the file list include a path that was never actually written.
    good = tmp_path / "Good.php"
    good.write_text("<?php\nclass Good {}\n")
    vanished = tmp_path / "Vanished.php"

    monkeypatch.setattr("app.scanner.php_classes.walk_module_source_files", lambda root: [good, vanished])

    classes = catalog_module_classes(tmp_path)

    assert {c.class_name for c in classes} == {"Good"}


def test_parse_php_file_captures_a_class_extends_clause(tmp_path: Path):
    path = tmp_path / "Child.php"
    path.write_text("<?php\nclass Child extends ParentClass {}\n")

    entries = parse_php_file(path, "Child.php")

    assert len(entries) == 1
    assert entries[0].extends == ["ParentClass"]


def test_parse_php_file_captures_simple_string_const_declarations(tmp_path: Path):
    path = tmp_path / "Foo.php"
    path.write_text(
        "<?php\n"
        "class Foo {\n"
        "    const ADMIN_RESOURCE = 'Vendor_Module::foo';\n"
        "    const EMPTY = '';\n"
        "    const NOT_A_STRING = 1;\n"
        "    const NOT_RESOLVABLE = self::ADMIN_RESOURCE;\n"
        "}\n"
    )

    entries = parse_php_file(path, "Foo.php")

    assert len(entries) == 1
    assert entries[0].consts == {"ADMIN_RESOURCE": "Vendor_Module::foo", "EMPTY": ""}


def test_string_literal_value_returns_none_for_a_non_string_node():
    # _extract_consts's own value_node search already filters for
    # type in ("string", "encapsed_string") before ever calling this, so a
    # non-string node never reaches _string_literal_value through that path
    # -- this exercises its own documented contract directly, as the
    # general-purpose helper its docstring describes it as.
    source = b"<?php\nconst FOO = 1;\n"
    tree = parse_php(source)
    (integer_node,) = find_all(tree.root_node, {"integer"})

    assert _string_literal_value(integer_node, source) is None


def test_string_literal_value_returns_none_for_a_string_with_an_escape_sequence():
    # An escape sequence inside an encapsed_string splits it into more than
    # one string_content child (confirmed empirically: `"Hello\n World"`
    # parses as ['"', 'string_content', 'escape_sequence', 'string_content',
    # '"']) -- not a plain literal this function can resolve, same
    # "unresolvable without evaluating an expression" case as interpolation.
    source = b'<?php\nconst FOO = "Hello\\n World";\n'
    tree = parse_php(source)
    (string_node,) = find_all(tree.root_node, {"encapsed_string"})

    assert _string_literal_value(string_node, source) is None


def test_parse_php_file_captures_abstract_modifier(tmp_path: Path):
    path = tmp_path / "Foo.php"
    path.write_text("<?php\nabstract class Foo {}\nclass Bar {}\n")

    entries = {e.class_name: e for e in parse_php_file(path, "Foo.php")}

    assert entries["Foo"].is_abstract is True
    assert entries["Bar"].is_abstract is False


def test_parse_php_file_delegates_to_parse_php_source_with_the_files_bytes(tmp_path: Path):
    path = tmp_path / "Foo.php"
    path.write_text("<?php\nclass Foo {}\n")

    via_file = parse_php_file(path, "Foo.php")
    via_source = parse_php_source(path.read_bytes(), "Foo.php")

    assert via_file == via_source


def test_catalog_module_classes_parses_from_in_memory_sources_without_touching_disk(tmp_path: Path, monkeypatch):
    """Orchestrator.py reads every file's bytes once (read_module_source_files)
    and passes them here instead of letting this function re-read from disk --
    the whole point of the `sources` parameter. Confirm it never calls
    walk_module_source_files (the disk-walking entry point) when sources are
    given."""

    def _must_not_walk(root):
        raise AssertionError("must not walk the filesystem when sources is given")

    monkeypatch.setattr("app.scanner.php_classes.walk_module_source_files", _must_not_walk)

    sources = {
        "Foo.php": b"<?php\nclass Foo {}\n",
        "Bar.php": b"<?php\nclass Bar extends Foo {}\n",
    }
    classes = catalog_module_classes(tmp_path, sources)

    by_name = {c.class_name: c for c in classes}
    assert by_name["Foo"].file_path == "Foo.php"
    assert by_name["Bar"].file_path == "Bar.php"
    assert by_name["Bar"].extends == ["Foo"]


def test_catalog_module_classes_with_sources_tolerates_garbage_bytes_in_one_file(tmp_path: Path):
    """tree-sitter is error-tolerant by design (ERROR nodes, never an
    exception) and node_text() decodes with errors="replace" -- so malformed
    content doesn't actually raise SyntaxError/UnicodeDecodeError in
    practice. This just confirms it degrades to "no classes found" for that
    one file rather than ever crashing the whole batch."""
    sources = {"Good.php": b"<?php\nclass Good {}\n", "Broken.php": b"\xff\xfe not valid php or utf-8"}

    classes = catalog_module_classes(tmp_path, sources)

    assert {c.class_name for c in classes} == {"Good"}


def test_catalog_module_classes_with_an_empty_sources_dict_parses_nothing(tmp_path: Path, monkeypatch):
    def _must_not_walk(root):
        raise AssertionError("must not walk the filesystem when sources (even empty) is given")

    monkeypatch.setattr("app.scanner.php_classes.walk_module_source_files", _must_not_walk)

    assert catalog_module_classes(tmp_path, {}) == []


def test_entry_to_dict_and_back_round_trips_a_parsed_class(tmp_path: Path):
    path = tmp_path / "Foo.php"
    path.write_text(
        "<?php\nabstract class Foo extends Base implements Iface {\n"
        "    const ADMIN_RESOURCE = 'Vendor_Module::foo';\n"
        "    public function bar() {}\n"
        "}\n"
    )
    (entry,) = parse_php_file(path, "Foo.php")

    round_tripped = entry_from_dict(entry_to_dict(entry))

    assert round_tripped == entry
