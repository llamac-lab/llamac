use glob::glob;
use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_src_dir = crate_dir.join("../llamacpp");
    let llama_build_dir = llama_src_dir.join("build");
    let in_ci = std::env::var("CI").is_ok();

    if in_ci {
        // nuke if in ci
        let _ = fs::remove_dir_all(&llama_build_dir); // Kill previous CMake cache
        fs::create_dir_all(&llama_build_dir).expect("Failed to recreate build dir");
    }

    fs::create_dir_all(&llama_build_dir).expect("Failed to create build dir");

    // Configure
    let mut cmake_args = vec![
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DDBUILD_SHARED_LIBS=OFF",
        "-DLLAMA_STANDALONE=ON",
        "-DGGML_THREADS=ON",
    ];

    if in_ci {
        cmake_args.push("-DGGML_CUDA=OFF");
    } else {
        cmake_args.push("-DGGML_CUDA=ON");
        cmake_args.push("-DGGML_CUDA_FORCE_CUBLAS=ON");
    }

    cmake_args.push(llama_src_dir.to_str().unwrap());

    let status = Command::new("cmake")
        .current_dir(&llama_build_dir)
        .args(&cmake_args)
        .status()
        .expect("Failed to run cmake");

    if !status.success() {
        panic!("cmake failed");
    }

    // Build
    let status = Command::new("make")
        .arg(format!("-j{}", num_cpus::get()))
        .current_dir(&llama_build_dir)
        .status()
        .expect("Failed to run make");

    if !status.success() {
        panic!("make failed");
    }

    // Output paths
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile = env::var("PROFILE").unwrap();
    let target_dir = crate_dir.join("target").join(&profile);

    // folders to search for .so
    let so_dirs = [
        llama_build_dir.join("bin"),
        llama_build_dir.join("lib"),
        llama_build_dir.clone(), // just in case
    ];

    for so_dir in &so_dirs {
        let pattern_str = format!("{}/*.so", so_dir.to_str().unwrap());
        eprintln!("[debug] .so glob pattern: {pattern_str}");

        for entry in glob(&pattern_str).expect("Failed to read glob pattern") {
            match entry {
                Ok(path) => {
                    let filename = path.file_name().unwrap();

                    let dest_out = out_dir.join(filename);
                    fs::copy(&path, &dest_out)
                        .unwrap_or_else(|e| panic!("Failed to copy {path:?} to {dest_out:?}: {e}"));

                    let dest_target = target_dir.join(filename);
                    fs::copy(&path, &dest_target).unwrap_or_else(|e| {
                        panic!("Failed to copy {path:?} to {dest_target:?}: {e}")
                    });

                    println!("cargo:rustc-link-search=native={}", out_dir.display());
                    println!(
                        "cargo:rustc-link-lib=dylib={}",
                        filename
                            .to_str()
                            .unwrap()
                            .trim_start_matches("lib")
                            .trim_end_matches(".so")
                    );
                }
                Err(e) => eprintln!("Glob error: {e:?}"),
            }
        }
    }
}
