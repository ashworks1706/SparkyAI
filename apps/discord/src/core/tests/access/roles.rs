//! Role names and permissions a member carries to the engine.

use std::collections::HashMap;

use serenity::all::{Permissions, RoleId};

use crate::access::roles::{authorized_roles, can_write, member_permissions};
use crate::core::config::WRITE_CAPABILITY;

#[test]
fn discord_management_permissions_grant_write_access() {
    assert!(can_write(Permissions::MANAGE_GUILD));
    assert!(can_write(Permissions::ADMINISTRATOR));
    assert!(!can_write(Permissions::MANAGE_MESSAGES));
}

#[test]
fn a_guild_role_cannot_impersonate_the_write_capability() {
    let named = vec!["students".to_owned(), WRITE_CAPABILITY.to_owned()];
    assert_eq!(
        authorized_roles(
            named.clone(),
            Some(Permissions::MANAGE_MESSAGES),
            WRITE_CAPABILITY
        ),
        vec!["students".to_owned()]
    );
    assert_eq!(
        authorized_roles(named, Some(Permissions::MANAGE_GUILD), WRITE_CAPABILITY),
        vec!["students".to_owned(), WRITE_CAPABILITY.to_owned()]
    );
    assert_eq!(
        authorized_roles(vec!["students".to_owned()], None, WRITE_CAPABILITY),
        vec!["students".to_owned()]
    );
}

#[test]
fn permissions_are_the_union_of_everyone_and_held_roles() {
    let everyone = RoleId::new(1);
    let mods = RoleId::new(2);
    let admins = RoleId::new(3);
    let bits = HashMap::from([
        (everyone, Permissions::SEND_MESSAGES),
        (mods, Permissions::MANAGE_MESSAGES),
        (admins, Permissions::MANAGE_GUILD),
    ]);

    let plain = member_permissions(everyone, &[], &bits, false);
    assert_eq!(plain, Permissions::SEND_MESSAGES);
    assert!(!can_write(plain));

    let moderator = member_permissions(everyone, &[mods], &bits, false);
    assert!(moderator.contains(Permissions::SEND_MESSAGES | Permissions::MANAGE_MESSAGES));
    assert!(!can_write(moderator));

    let admin = member_permissions(everyone, &[mods, admins, RoleId::new(99)], &bits, false);
    assert!(
        can_write(admin),
        "a held role grants write; an unknown role adds nothing"
    );

    let owner = member_permissions(everyone, &[], &bits, true);
    assert!(can_write(owner), "the guild owner writes without any role");
}
