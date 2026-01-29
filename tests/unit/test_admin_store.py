from __future__ import annotations

from app.discord.admin_store import AdminStore


def test_admin_store_bootstrap_and_roles(tmp_path):
    db_path = tmp_path / "admins.sqlite3"
    store = AdminStore(db_path, master_superadmins=[1], debug_guild_ids=[])
    store.init_schema()
    store.bootstrap_superadmins()

    assert store.is_superadmin(1)
    assert store.is_admin(1)

    store.upsert_admin(2, "admin")
    assert store.is_admin(2)
    assert not store.is_superadmin(2)

    store.upsert_admin(2, "superadmin")
    assert store.is_superadmin(2)
    assert store.has_role(2, "admin") is True
    assert store.has_role(2, "admin") is True


def test_admin_store_remove(tmp_path):
    db_path = tmp_path / "admins.sqlite3"
    store = AdminStore(db_path, master_superadmins=[1], debug_guild_ids=[])
    store.init_schema()
    store.bootstrap_superadmins()

    store.upsert_admin(2, "admin")
    assert store.remove_admin(2) is True
    assert store.get_admin(2) is None

    # master superadmin should not be removable
    assert store.remove_admin(1) is False


def test_admin_store_migrates_legacy_table(tmp_path):
    db_path = tmp_path / "admins.sqlite3"
    store = AdminStore(db_path, master_superadmins=[], debug_guild_ids=[])
    store.init_schema()
    with store._connect() as conn:
        conn.execute("INSERT INTO admins (user_id, role) VALUES (?, ?)", (7, "admin"))
        conn.commit()
    store.init_schema()
    assert store.has_role(7, "admin")


def test_admin_store_list_admins(tmp_path):
    db_path = tmp_path / "admins.sqlite3"
    store = AdminStore(db_path, master_superadmins=[], debug_guild_ids=[])
    store.init_schema()
    store.add_role(2, "admin")
    store.add_role(1, "superadmin")
    out = store.list_admins()
    assert [(r.user_id, r.role) for r in out] == [(1, "superadmin"), (2, "admin")]


def test_admin_store_debug_guilds(tmp_path):
    db_path = tmp_path / "admins.sqlite3"
    store = AdminStore(db_path, master_superadmins=[], debug_guild_ids=[3, 2])
    store.init_schema()
    store.bootstrap_debug_guilds()
    assert store.list_debug_guilds() == [2, 3]
    store.add_debug_guild(4)
    assert store.list_debug_guilds() == [2, 3, 4]
    assert store.remove_debug_guild(3) is True
    assert store.list_debug_guilds() == [2, 4]


def test_admin_store_migrates_legacy_table(tmp_path):
    db_path = tmp_path / "admins.sqlite3"
    store = AdminStore(db_path, master_superadmins=[], debug_guild_ids=[])
    store.init_schema()
    with store._connect() as conn:
        conn.execute("INSERT INTO admins (user_id, role) VALUES (?, ?)", (7, "admin"))
        conn.commit()
    store.init_schema()
    assert store.has_role(7, "admin")
