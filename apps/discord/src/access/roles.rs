//! Role names and permission bits a member carries into the engine.

use std::collections::HashMap;

use serenity::all::{Permissions, RoleId};

/// Whether the Discord permissions of a member let them ask for write-side tools.
pub(crate) fn can_write(permissions: Permissions) -> bool {
    permissions.intersects(Permissions::MANAGE_GUILD | Permissions::ADMINISTRATOR)
}

/// Guild role plus capability when Discord permissions grant it; a role name alone grants nothing.
pub(crate) fn authorized_roles(
    names: impl IntoIterator<Item = String>,
    permissions: Option<Permissions>,
    capability: &str,
) -> Vec<String> {
    let mut roles: Vec<String> = names.into_iter().filter(|n| n != capability).collect();
    if permissions.is_some_and(can_write) {
        roles.push(capability.to_owned());
    }
    roles
}

/// Permissions of a member: all bits for the owner, else union of everyone role and roles held.
pub(crate) fn member_permissions(
    everyone: RoleId,
    held: &[RoleId],
    bits: &HashMap<RoleId, Permissions>,
    owner: bool,
) -> Permissions {
    if owner {
        return Permissions::all();
    }
    std::iter::once(&everyone)
        .chain(held)
        .filter_map(|id| bits.get(id))
        .fold(Permissions::empty(), |acc, p| acc | *p)
}
