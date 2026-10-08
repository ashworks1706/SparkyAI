//! The egress network and proxy a sandbox reaches the public internet through.

use crate::core::types::tools::sandbox::SandboxError;
use crate::runtime::tools::sandbox::container::{ContainerSandbox, LABEL};

impl ContainerSandbox {
    /// Makes the egress network and its proxy ready, when egress is on.
    pub async fn prepare_egress(&self) -> Result<(), SandboxError> {
        let Some(egress) = &self.limits.egress else {
            return Ok(());
        };
        let internal = self
            .inspect(&[
                "network",
                "inspect",
                "--format",
                "{{.Internal}}",
                &egress.network,
            ])
            .await;
        match internal.as_deref() {
            Some("true") => {}
            Some(other) => {
                return Err(SandboxError::Runtime(format!(
                    "network {} is not internal (Internal={other}); remove it or name another \
                     in sandbox.egress_network",
                    egress.network
                )));
            }
            None => {
                self.runtime(&["network", "create", "--internal", &egress.network])
                    .await?;
            }
        }
        if !self.running(&egress.proxy_name).await {
            self.runtime_ok(&["rm", "--force", &egress.proxy_name])
                .await;
            self.runtime(&[
                "run",
                "--detach",
                "--label",
                LABEL,
                "--name",
                &egress.proxy_name,
                "--restart",
                "unless-stopped",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--memory",
                "128m",
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                "/var/run/squid",
                "--tmpfs",
                "/var/log/squid",
                "--tmpfs",
                "/var/spool/squid",
                &egress.proxy_image,
            ])
            .await?;
        }
        let joined = self
            .inspect(&[
                "inspect",
                "--format",
                &format!(
                    "{{{{index .NetworkSettings.Networks {:?}}}}}",
                    egress.network
                ),
                &egress.proxy_name,
            ])
            .await;
        if joined.is_none_or(|j| j == "<nil>" || j.is_empty()) {
            self.runtime(&["network", "connect", &egress.network, &egress.proxy_name])
                .await?;
        }
        Ok(())
    }
}
