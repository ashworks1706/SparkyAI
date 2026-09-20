import { useEffect, useRef } from "react";
import * as THREE from "three";

const MAROON = 0x8c1d40;
const GOLD = 0xffc627;

/** Proportions of the logo file, so the plane never squashes the dragon. */
const DRAGON = {
  src: "/brand/sparkyai-logo.png",
  width: 717,
  height: 779,
  tall: 3.4,
};

/** One shape orbiting the dragon, and the clock offsets that keep it out of lockstep. */
type Satellite = {
  mesh: THREE.Mesh;
  speed: number;
  offset: number;
  amount: number;
  base: THREE.Vector3;
};

const glass = (color: number, transmission: number, opacity: number) =>
  new THREE.MeshPhysicalMaterial({
    color,
    roughness: 0.1,
    metalness: 0.28,
    transmission,
    thickness: 0.55,
    transparent: true,
    opacity,
  });

/** The shapes that float around the dragon. Low segment counts: these read as glass, not geometry. */
const satellites = (): Satellite[] => {
  const specs: [
    THREE.BufferGeometry,
    THREE.Material,
    [number, number, number],
    number,
    number,
    number,
  ][] = [
    [
      new THREE.IcosahedronGeometry(0.42, 0),
      glass(GOLD, 0.55, 0.92),
      [2.05, 0.9, -1.2],
      0.7,
      0,
      0.18,
    ],
    [
      new THREE.IcosahedronGeometry(0.28, 0),
      glass(MAROON, 0.4, 0.9),
      [-1.95, -0.9, -1],
      0.52,
      2.1,
      0.22,
    ],
    [
      new THREE.TorusGeometry(0.36, 0.1, 16, 48),
      new THREE.MeshPhysicalMaterial({
        color: GOLD,
        roughness: 0.2,
        metalness: 0.5,
        clearcoat: 0.8,
      }),
      [1.6, -1.3, 0.9],
      0.85,
      4.2,
      0.14,
    ],
    [
      new THREE.OctahedronGeometry(0.3, 0),
      glass(0xffffff, 0.8, 0.75),
      [-1.7, 1.25, 0.8],
      0.6,
      1.1,
      0.2,
    ],
  ];
  return specs.map(([geometry, material, at, speed, offset, amount]) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...at);
    return { mesh, speed, offset, amount, base: mesh.position.clone() };
  });
};

/**
 * Sparky at the centre with glass shapes around him, the whole cluster leaning toward the pointer.
 *
 * Written against three directly rather than a React renderer: the scene is built once and never
 * reacts to a prop, so a reconciler would only add a dependency that has to track React releases.
 */
const Scene3D = () => {
  const holder = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = holder.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.z = 6;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.domElement.style.pointerEvents = "none";
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 2.1);
    key.position.set(4, 5, 5);
    const warm = new THREE.DirectionalLight(GOLD, 1.1);
    warm.position.set(-5, -2, 2);
    const rim = new THREE.PointLight(MAROON, 12, 9);
    rim.position.set(0, 0, 3);
    scene.add(key, warm, rim);

    const cluster = new THREE.Group();
    scene.add(cluster);

    const orbiting = satellites();
    for (const item of orbiting) cluster.add(item.mesh);

    // The plane swings through a few degrees rather than turning: seen edge on it would vanish.
    const texture = new THREE.TextureLoader().load(DRAGON.src);
    texture.colorSpace = THREE.SRGBColorSpace;
    const dragon = new THREE.Mesh(
      new THREE.PlaneGeometry(
        (DRAGON.width / DRAGON.height) * DRAGON.tall,
        DRAGON.tall,
      ),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true }),
    );
    cluster.add(dragon);

    const pointer = new THREE.Vector2();
    const onPointer = (event: PointerEvent) => {
      const box = mount.getBoundingClientRect();
      pointer.set(
        ((event.clientX - box.left) / box.width) * 2 - 1,
        -(((event.clientY - box.top) / box.height) * 2 - 1),
      );
    };
    window.addEventListener("pointermove", onPointer, { passive: true });

    const resize = () => {
      const { clientWidth: w, clientHeight: h } = mount;
      if (w === 0 || h === 0) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    const clock = new THREE.Clock();
    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      const t = clock.getElapsedTime();

      cluster.rotation.y += (pointer.x * 0.22 - cluster.rotation.y) * 0.04;
      cluster.rotation.x += (-pointer.y * 0.16 - cluster.rotation.x) * 0.04;

      dragon.rotation.y = Math.sin(t * 0.35) * 0.3;
      dragon.rotation.z = Math.sin(t * 0.25) * 0.05;
      dragon.position.y = Math.sin(t * 0.5) * 0.14;

      for (const item of orbiting) {
        const at = t * item.speed + item.offset;
        item.mesh.position.y = item.base.y + Math.sin(at) * item.amount;
        item.mesh.rotation.x = Math.sin(at * 0.5) * 0.4;
        item.mesh.rotation.z = Math.cos(at * 0.4) * 0.4;
      }
      renderer.render(scene, camera);
    };
    tick();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      window.removeEventListener("pointermove", onPointer);
      renderer.domElement.remove();
      renderer.dispose();
      texture.dispose();
      scene.traverse((node) => {
        if (!(node instanceof THREE.Mesh)) return;
        node.geometry.dispose();
        const material = node.material as THREE.Material | THREE.Material[];
        if (Array.isArray(material)) material.forEach((m) => m.dispose());
        else material.dispose();
      });
    };
  }, []);

  return <div ref={holder} className="h-full w-full" data-testid="scene-3d" />;
};

export default Scene3D;
