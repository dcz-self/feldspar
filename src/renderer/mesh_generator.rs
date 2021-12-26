use crate::{
    prelude::{
        ambient_sdf_array, ArrayMaterial, DirtyChunks, MaterialLayer, MaterialVoxel, SdfVoxelMap,
        SmoothVoxelPbrBundle, ThreadLocalResource, VoxelType,
    },
    BevyState,
};

use building_blocks::{
    mesh::*,
    prelude::{
        Array3x1, ChunkKey3, IndexedArray, IsEmpty, Local, Point3i, Stride, TransformMap,
    },
    storage::access_traits::*,
    storage::SmallKeyHashMap,
};

use bevy::{
    asset::prelude::*,
    ecs,
    prelude::*,
    render::{
        mesh::{Indices, VertexAttributeValues},
        pipeline::PrimitiveTopology,
    },
    tasks::ComputeTaskPool,
};
use std::cell::RefCell;
use std::cmp;


/// Do not render anything above this level.
/// It could also be f32, to operate on world coords,
/// but it's easier to use voxel coords.
#[derive(PartialEq, Clone, Copy)]
pub struct MeshCutoff(pub i32);

impl MeshCutoff {
    pub fn nothing() -> Self {
        MeshCutoff(i32::MAX)
    }
}

impl Default for MeshCutoff {
    fn default() -> Self {
        MeshCutoff(i32::MAX)
    }
}

// TODO: make a collection of textures for different attributes (albedo, normal, metal, rough, emmisive, etc)
#[derive(Default)]
pub struct MeshMaterial(pub Handle<ArrayMaterial>);

/// Generates smooth meshes for voxel chunks. When a chunk becomes dirty, its old mesh is replaced with a newly generated one.
///
/// **NOTE**: Expects the `MeshMaterial` resource to exist before running. You should specify the state `S` when things have
/// loaded and this resource exists.
pub struct MeshGeneratorPlugin<S> {
    update_state: S,
}

impl<S> MeshGeneratorPlugin<S> {
    pub fn new(update_state: S) -> Self {
        Self { update_state }
    }
}

impl<S: BevyState> Plugin for MeshGeneratorPlugin<S> {
    fn build(&self, app: &mut AppBuilder) {
        app
            .insert_resource(ChunkMeshes::default())
            .insert_resource(MeshCutoff::nothing())
            .add_system_set(
                SystemSet::on_update(self.update_state.clone())
                    .with_system(mesh_generator_system.system()),
            );
    }
}

#[derive(Default)]
pub struct ChunkMeshes {
    // Map from chunk key to mesh entity.
    entities: SmallKeyHashMap<ChunkKey3, Entity>,
    used_cutoff: MeshCutoff,
}

/// Generates new meshes for all dirty chunks.
fn mesh_generator_system(
    mut commands: Commands,
    pool: Res<ComputeTaskPool>,
    voxel_map: Res<SdfVoxelMap>,
    dirty_chunks: Res<DirtyChunks>,
    cutoff_height: Res<MeshCutoff>,
    local_mesh_buffers: ecs::system::Local<ThreadLocalMeshBuffers>,
    mesh_material: Res<MeshMaterial>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunk_meshes: ResMut<ChunkMeshes>,
) {
    let all_chunks =
        if *cutoff_height == chunk_meshes.used_cutoff {
            None
        } else {
            println!("cutoff new: {} old {}", cutoff_height.0, chunk_meshes.used_cutoff.0);
            let keys = voxel_map.voxels.get_chunk_keys();
            Some(DirtyChunks::new(keys.as_slice()))
        };

    let new_chunk_meshes = generate_mesh_for_each_chunk(
        &*voxel_map,
        all_chunks.as_ref().unwrap_or(&*dirty_chunks),
        &*cutoff_height,
        &*local_mesh_buffers,
        &*pool,
    );
    chunk_meshes.used_cutoff = *cutoff_height;
    for (chunk_key, item) in new_chunk_meshes.into_iter() {
        let old_mesh = if let Some((mesh, material_counts)) = item {
            log::debug!("Creating chunk mesh for {:?}", chunk_key);
            chunk_meshes.entities.insert(
                chunk_key,
                commands
                    .spawn_bundle(create_voxel_mesh_bundle(
                        mesh,
                        material_counts,
                        mesh_material.0.clone(),
                        &mut *meshes,
                    ))
                    .id(),
            )
        } else {
            chunk_meshes.entities.remove(&chunk_key)
        };
        if let Some(old_mesh) = old_mesh {
            commands.entity(old_mesh).despawn();
        }
    }
}

fn generate_mesh_for_each_chunk(
    voxel_map: &SdfVoxelMap,
    dirty_chunks: &DirtyChunks,
    cutoff_height: &MeshCutoff,
    local_mesh_buffers: &ThreadLocalMeshBuffers,
    pool: &ComputeTaskPool,
) -> Vec<(ChunkKey3, Option<(PosNormMesh, Vec<[u8; 4]>)>)> {
    pool.scope(|s| {
        for chunk_min in dirty_chunks.dirty_chunk_mins().iter().cloned() {
            let chunk_key = ChunkKey3::new(0, chunk_min);
            s.spawn(async move {
                let padded_chunk_extent = padded_greedy_quads_chunk_extent(
                    &voxel_map
                        .voxels
                        .indexer
                        .extent_for_chunk_with_min(chunk_min),
                );

                let mesh_tls = local_mesh_buffers.get();
                let mut mesh_buffers = mesh_tls
                    .get_or_create_with(|| {
                        RefCell::new(MeshBuffers {
                            padded_chunk: Array3x1::fill(padded_chunk_extent, VoxelType(0)),
                            greedy_quads_buffer: GreedyQuadsBuffer::new(
                                padded_chunk_extent,
                                RIGHT_HANDED_Y_UP_CONFIG.quad_groups(),
                            ),
                        })
                    })
                    .borrow_mut();

                let MeshBuffers {
                    padded_chunk,
                    greedy_quads_buffer,
                } = &mut *mesh_buffers;

                padded_chunk.set_minimum(padded_chunk_extent.minimum);
                let lod_view = voxel_map.voxels.lod_view(0);
                //let voxels = TransformIndexMap::new(&lod_view, |(type_, _dist), _coord| type_);
                let voxels = TransformMap::new(&lod_view, |(type_, _dist)| type_);

                // Poor man's filter
                let mut truncated_chunk_extent = padded_chunk_extent.clone();
                *truncated_chunk_extent.shape.y_mut()
                    = cmp::max(
                        cmp::min(
                            truncated_chunk_extent.shape.y() + truncated_chunk_extent.minimum.y(),
                            cutoff_height.0
                        )
                        - truncated_chunk_extent.minimum.y(),
                        0,
                    );

                if truncated_chunk_extent.shape.y() <= 0 {
                    return (chunk_key, None);
                }

                println!("Padded {:?}", padded_chunk_extent);
                println!("Truncaeted {:?}", truncated_chunk_extent);
                copy_extent(
                    &truncated_chunk_extent,
                    //&padded_chunk_extent,
                    &voxels,
                    padded_chunk,
                );

                let padded_greedy_chunk = padded_chunk;

                greedy_quads(
                    padded_greedy_chunk,
                    &padded_chunk_extent,
                    &mut *greedy_quads_buffer,
                );

                let mut mesh = PosNormMesh::default();
                let mut materials = Vec::new();
                for group in greedy_quads_buffer.quad_groups.iter() {
                    for quad in group.quads.iter() {
                        group.face.add_quad_to_pos_norm_mesh(&quad, 1.0, &mut mesh);
                        let material = to_material(&voxel_map, padded_greedy_chunk.get(quad.minimum));
                        // Four copies of the material, for each corner of the quad
                        materials.extend_from_slice(&[material, material, material, material]);
                    }
                }

                (chunk_key, Some((mesh.clone(), materials)))
            })
        }
    })
}

fn to_material(world: &SdfVoxelMap, v: VoxelType) -> [u8; 4] {
    let mut materials = [0; 4];
    let info = world.palette.get_voxel_type_info(v);
    materials[info.material().0 as usize] = 1;
    materials
}

/// Uses a kernel to count the adjacent materials for each surface point. This is necessary because we used dual contouring to
/// construct the mesh, so a given vertex has 8 adjacent voxels, some of which may be empty. This also assumes that the material
/// layer can only be one of 0..4.
fn count_adjacent_materials<A, V>(voxels: &A, surface_strides: &[Stride]) -> Vec<[u8; 4]>
where
    A: IndexedArray<[i32; 3]> + Get<Stride, Item = V>,
    V: IsEmpty + MaterialVoxel,
{
    let mut corner_offsets = [Stride(0); 8];
    voxels.strides_from_local_points(
        &Local::localize_points_array(&Point3i::CUBE_CORNER_OFFSETS),
        &mut corner_offsets,
    );
    let mut material_counts = vec![[0; 4]; surface_strides.len()];
    for (stride, counts) in surface_strides.iter().zip(material_counts.iter_mut()) {
        for corner in corner_offsets.iter() {
            let corner_voxel = voxels.get(*stride + *corner);
            // Only add weights from non-empty voxels.
            if !corner_voxel.is_empty() {
                let material = corner_voxel.material();
                debug_assert!(material != MaterialLayer::NULL);
                counts[material.0 as usize] += 1;
            }
        }
    }

    material_counts
}

// ThreadLocal doesn't let you get a mutable reference, so we need to use RefCell. We lock this down to only be used in this
// module as a Local resource, so we know it's safe.
type ThreadLocalMeshBuffers = ThreadLocalResource<RefCell<MeshBuffers>>;

pub struct MeshBuffers {
    //surface_nets_buffer: SurfaceNetsBuffer,
    greedy_quads_buffer: GreedyQuadsBuffer,
    padded_chunk: Array3x1<VoxelType>,
}

fn create_voxel_mesh_bundle(
    mesh: PosNormMesh,
    material_counts: Vec<[u8; 4]>,
    material: Handle<ArrayMaterial>,
    meshes: &mut Assets<Mesh>,
) -> SmoothVoxelPbrBundle {
    assert_eq!(mesh.positions.len(), mesh.normals.len());
    assert_eq!(mesh.positions.len(), material_counts.len());

    let mut render_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    render_mesh.set_attribute(
        "Vertex_Position",
        VertexAttributeValues::Float3(mesh.positions),
    );
    render_mesh.set_attribute("Vertex_Normal", VertexAttributeValues::Float3(mesh.normals));
    render_mesh.set_attribute(
        "Vertex_MaterialWeights",
        VertexAttributeValues::Uint(
            material_counts
                .into_iter()
                .map(|c| {
                    (c[0] as u32) | (c[1] as u32) << 8 | (c[2] as u32) << 16 | (c[3] as u32) << 24
                })
                .collect(),
        ),
    );
    render_mesh.set_indices(Some(Indices::U32(mesh.indices)));

    SmoothVoxelPbrBundle {
        mesh: meshes.add(render_mesh),
        material,
        ..Default::default()
    }
}
