from importlib.metadata import distribution

import pytest

from mrnax.constraints.fto_guard import count_component_gpl_dependencies


def installed_count(tmp_path, monkeypatch, notices, expression=None):
    info = tmp_path / "mixed_notice-1.0.dist-info"
    (info / "licenses").mkdir(parents=True)
    fields = ["Metadata-Version: 2.4", "Name: mixed-notice", "Version: 1.0"]
    if expression:
        fields.append(f"License-Expression: {expression}")
    for index, text in enumerate(notices):
        name = f"COPYING.{index}"
        (info / "licenses" / name).write_text(text)
        fields.append(f"License-File: {name}")
    (info / "METADATA").write_text("\n".join(fields) + "\n\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    return count_component_gpl_dependencies("mixed-notice", "MIT", "n/a")


HELPER = "Bundled helper only\nSPDX-License-Identifier: MIT"
MAIN_GPL = "The main library is licensed exclusively under the GNU General Public License version 3 only."
HEADER = "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007"


@pytest.mark.parametrize(("notices", "expression", "expected"), [
    ([MAIN_GPL + "\n" + HELPER], None, 1),
    ([HELPER + "\n" + MAIN_GPL], None, 1),
    ([HELPER, MAIN_GPL], None, 1),
    ([HELPER + "\nSPDX-License-Identifier: AGPL-3.0-only"], None, 1),
    ([HELPER + "\n" + HEADER], None, 1),
    (["The main library is licensed under the MIT License.\n" + HELPER], None, 0),
    (["This program can be redistributed under the terms of the GPL.\n" + HELPER], None, 1),
    (["Previously distributed under the GPL, the library now uses MIT.\n" + HELPER], None, 0),
    (["Compatible with applications licensed under the GPL.\n" + MAIN_GPL], None, 1),
    (["If you combine this software with software that is licensed under the GPLv2."], None, 0),
    (["This library ships with software that is licensed under the GPLv2."], None, 1),
    (["LicenseRef-Proprietary-Custom"], None, 1),
    ([HEADER, HELPER], "MIT OR GPL-3.0-only", 0),
    ([HEADER, HELPER], "MIT", 1),
    ([HEADER], "(MIT)OR(GPL-3.0-only)", 0),
    ([HEADER, "The bundled parser is licensed exclusively under the GNU Affero General Public License."],
     "MIT OR GPL-3.0-only", 1),
    ([HEADER, "Bundled parser only\nSPDX-License-Identifier: GPL-3.0-only"], "MIT OR GPL-3.0-only", 1),
    (["## License: GNU General Public License (GPL), version 3"], None, 1),
    (["The main library is GPL-3.0-only.\n" + HELPER], None, 1),
    (["The main library is GPL-compatible.\n" + HELPER], None, 0),
    ([HELPER + "\nThe license for the main library is GPL-3.0-only."], None, 1),
    (["Versions prior to 2.0 were licensed under the GPL. This version is licensed under MIT."], None, 0),
    (["At your option, you may use this entire package under either the MIT License or the GNU General Public License.\n"
      + HEADER], None, 0),
    ([HELPER + "\nThis entire package is dual-licensed under the MIT License or the GNU General Public License.\n"
      + HEADER], None, 0),
    (["The bundled helper is dual-licensed under the MIT License or GPL-3.0-only.\n" + MAIN_GPL], None, 1),
])
def test_all_file_grants_contribute_to_package_license(tmp_path, monkeypatch, notices, expression, expected):
    assert installed_count(tmp_path, monkeypatch, notices, expression) == expected


def test_shipped_psf_license_without_optional_metadata_expression(tmp_path, monkeypatch):
    dist = distribution("typing_extensions")
    file = next(path for path in dist.files if path.name == "LICENSE" and ".dist-info/" in str(path))
    notice = dist.locate_file(file).read_text()
    assert "PYTHON SOFTWARE FOUNDATION LICENSE" in notice and "GPL-compatible" in notice
    assert installed_count(tmp_path, monkeypatch, [notice]) == 0
    assert installed_count(tmp_path / "additional", monkeypatch, [notice, MAIN_GPL]) == 1
