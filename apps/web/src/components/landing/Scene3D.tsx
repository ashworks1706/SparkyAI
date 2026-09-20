import { Canvas, useFrame, useLoader } from "@react-three/fiber";
import { useRef } from "react";
import type { ReactNode } from "react";
import { TextureLoader } from "three";
import type { Group, Mesh } from "three";

const MAROON = "#8C1D40";
const GOLD = "#FFC627";

/** Proportions of the logo file, so the plane never squashes the dragon. */
const DRAGON = {
  src: "/brand/sparkyai-logo.png",
  width: 717,
  height: 779,
  tall: 3.4,
};

/**
 * Bobs and turns whatever it holds. Each one runs off the shared clock with its own offset, so
 * the group never moves in lockstep.
 */
const Float = ({
  children,
  speed = 1,
  offset = 0,
  amount = 0.18,
}: {
  children: ReactNode;
  speed?: number;
  offset?: number;
  amount?: number;
}) => {
  const group = useRef<Group>(null);
  useFrame((state) => {
    if (!group.current) return;
    const t = state.clock.elapsedTime * speed + offset;
    group.current.position.y = Math.sin(t) * amount;
    group.current.rotation.x = Math.sin(t * 0.5) * 0.2;
    group.current.rotation.z = Math.cos(t * 0.4) * 0.2;
  });
  return <group ref={group}>{children}</group>;
};

/**
 * Sparky at the centre, on a plane carrying the logo.
 *
 * It swings through a few degrees rather than turning: a plane seen edge on is a line, so a full
 * rotation would lose the dragon twice a cycle. The material is unlit, because the artwork
 * already carries its own shading.
 */
const Dragon = () => {
  const mesh = useRef<Mesh>(null);
  const texture = useLoader(TextureLoader, DRAGON.src);
  const width = (DRAGON.width / DRAGON.height) * DRAGON.tall;

  useFrame((state) => {
    if (!mesh.current) return;
    const t = state.clock.elapsedTime;
    mesh.current.rotation.y = Math.sin(t * 0.35) * 0.3;
    mesh.current.rotation.z = Math.sin(t * 0.25) * 0.05;
    mesh.current.position.y = Math.sin(t * 0.5) * 0.14;
  });

  return (
    <mesh ref={mesh}>
      <planeGeometry args={[width, DRAGON.tall]} />
      <meshBasicMaterial map={texture} transparent toneMapped={false} />
    </mesh>
  );
};

/** The shapes around it. Low segment counts: these read as glass, not as geometry. */
const Satellites = () => (
  <>
    <Float speed={0.7} offset={0}>
      <mesh position={[2.05, 0.9, -1.2]}>
        <icosahedronGeometry args={[0.42, 0]} />
        <meshPhysicalMaterial
          color={GOLD}
          roughness={0.08}
          metalness={0.2}
          transmission={0.55}
          thickness={0.6}
          transparent
          opacity={0.92}
        />
      </mesh>
    </Float>
    <Float speed={0.52} offset={2.1} amount={0.22}>
      <mesh position={[-1.95, -0.9, -1]}>
        <icosahedronGeometry args={[0.28, 0]} />
        <meshPhysicalMaterial
          color={MAROON}
          roughness={0.1}
          metalness={0.3}
          transmission={0.4}
          thickness={0.5}
          transparent
          opacity={0.9}
        />
      </mesh>
    </Float>
    <Float speed={0.85} offset={4.2} amount={0.14}>
      <mesh position={[1.6, -1.3, 0.9]} rotation={[0.4, 0.2, 0]}>
        <torusGeometry args={[0.36, 0.1, 16, 48]} />
        <meshPhysicalMaterial
          color={GOLD}
          roughness={0.2}
          metalness={0.5}
          clearcoat={0.8}
        />
      </mesh>
    </Float>
    <Float speed={0.6} offset={1.1} amount={0.2}>
      <mesh position={[-1.7, 1.25, 0.8]}>
        <octahedronGeometry args={[0.3, 0]} />
        <meshPhysicalMaterial
          color="#ffffff"
          roughness={0.05}
          metalness={0.1}
          transmission={0.8}
          thickness={0.4}
          transparent
          opacity={0.75}
        />
      </mesh>
    </Float>
  </>
);

/** Tilts the whole cluster toward the pointer. One group, so it costs one transform a frame. */
const Parallax = ({ children }: { children: ReactNode }) => {
  const group = useRef<Group>(null);
  useFrame((state) => {
    if (!group.current) return;
    const { x, y } = state.pointer;
    group.current.rotation.y += (x * 0.22 - group.current.rotation.y) * 0.04;
    group.current.rotation.x += (-y * 0.16 - group.current.rotation.x) * 0.04;
  });
  return <group ref={group}>{children}</group>;
};

/**
 * The hero object. Rendered at a capped pixel ratio and with no shadow map: on a light page the
 * lighting carries the form, and shadows would cost more than they show.
 */
const Scene3D = () => (
  <Canvas
    camera={{ position: [0, 0, 6], fov: 42 }}
    dpr={[1, 1.75]}
    gl={{ antialias: true, alpha: true }}
    style={{ pointerEvents: "none" }}
  >
    <ambientLight intensity={1.1} />
    <directionalLight position={[4, 5, 5]} intensity={2.1} color="#ffffff" />
    <directionalLight position={[-5, -2, 2]} intensity={1.1} color={GOLD} />
    <pointLight
      position={[0, 0, 3]}
      intensity={12}
      color={MAROON}
      distance={9}
    />
    <Parallax>
      <Dragon />
      <Satellites />
    </Parallax>
  </Canvas>
);

export default Scene3D;
