import { useEffect, useRef } from "react";
import * as THREE from "three";

/** Proportions of the logo file, so the plane never squashes the dragon. */
const DRAGON = {
  src: "/brand/sparkyai-logo.png",
  width: 717,
  height: 779,
  tall: 3.4,
};

/**
 * Sparky, floating and leaning toward the pointer.
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
    const fill = new THREE.DirectionalLight(0xffffff, 1.1);
    fill.position.set(-5, -2, 2);
    scene.add(key, fill);

    const cluster = new THREE.Group();
    scene.add(cluster);

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
