import pytest

from mrnax.constraints.fto_guard import count_component_gpl_dependencies


@pytest.mark.parametrize(("fields", "notice", "location", "expected"), [
    ([], "SPDX-License-Identifier: GPL-3.0-only\n", "licenses", 1),
    ([], "SPDX-License-Identifier: (MIT OR GPL-3.0-only) AND Apache-2.0\n", "licenses", 0),
    ([], "/* SPDX-License-Identifier: MIT AND LGPL-2.1-only */\n", "licenses", 1),
    (["License: See accompanying files"], "GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3", "licenses", 1),
    (["Classifier: License :: OSI Approved"], "GNU LESSER GENERAL PUBLIC LICENSE\nVersion 3", "licenses", 1),
    ([], "GNU LIBRARY GENERAL PUBLIC LICENSE\nVersion 2, June 1991", "licenses", 1),
    (["License-Expression: "], "SPDX-License-Identifier: MIT", "licenses", 0),
    ([], "GNU GENERAL PUBLIC LICENSE\nVersion 2, June 1991", "", 1),
    ([], "The MIT License\nPermission is hereby granted, free of charge", "licenses", 0),
    (["License: MIT", "Classifier: License :: OSI Approved :: MIT License"],
     "MIT License\nPermission is hereby granted, free of charge.\n"
     "This library is compatible with applications licensed under the GNU General Public License.",
     "licenses", 0),
    (["Classifier: License :: OSI Approved :: MIT License"],
     "This library is distributed exclusively under the GNU Lesser General Public License.", "licenses", 1),
    ([], "Copyright 2026\nExampleLib is licensed under the GNU General Public License version 3.", "licenses", 1),
    ([], "This project is licensed under GNU GPL v3.", "licenses", 1),
    ([], "This program is free software: you can redistribute it and/or modify it\n"
     "under the terms of the GNU General Public License as published by the Free Software Foundation.",
     "licenses", 1),
    (["License-Expression: MIT OR GPL-3.0-only"], "GNU GENERAL PUBLIC LICENSE", "licenses", 0),
])
def test_installed_license_files(tmp_path, monkeypatch, fields, notice, location, expected):
    info = tmp_path / "example_license-1.0.dist-info"
    declared = "notices/terms.txt"
    path = info / location / declared
    path.parent.mkdir(parents=True)
    path.write_text(notice)
    # A second declared file can impose obligations independent of the first.
    (info / "licenses").mkdir(exist_ok=True)
    (info / "licenses/COPYING").write_text("SPDX-License-Identifier: MIT")
    metadata = ["Metadata-Version: 2.4", "Name: example-license", "Version: 1.0", *fields,
                "License-File: COPYING", f"License-File: {declared}"]
    (info / "METADATA").write_text("\n".join(metadata) + "\n\n")
    members = ("METADATA", "licenses/COPYING", str(path.relative_to(info)), "RECORD")
    (info / "RECORD").write_text("".join(f"{info.name}/{name},,\n" for name in members))
    monkeypatch.syspath_prepend(str(tmp_path))
    assert count_component_gpl_dependencies("example-license", "MIT", "n/a") == expected


@pytest.mark.parametrize("declared", ["../other-license", "/other-license", "missing"])
def test_unresolved_license_file_does_not_clear_dependency(tmp_path, monkeypatch, declared):
    info = tmp_path / "example_license-1.0.dist-info"
    info.mkdir()
    (tmp_path / "other-license").write_text("SPDX-License-Identifier: MIT")
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.4\nName: example-license\nVersion: 1.0\nLicense-File: {declared}\n\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert count_component_gpl_dependencies("example-license", "MIT", "n/a") == 1
