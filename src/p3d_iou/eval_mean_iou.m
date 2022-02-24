function [ious] = eval_mean_iou(exp_name, eval_split, voc_class, p3d_dir)
    addpath('./matUtils');
    addpath('./voxelization');

    grid_size = 32;
    exp_name
    basedir = fullfile(pwd, '..', '..');

    if strcmp(eval_split, 'train')
        csdm_dir = fullfile(basedir, '..', 'drcCodeRelease', 'cachedir', 'pascal', 'camera', voc_class, 'train');
    else
        csdm_dir = fullfile(basedir, 'cachedir', 'p3d_eval', 'modelVoxelsCs3d', voc_class);
    end
    csdm_files = getFileNamesFromDirectory(csdm_dir, 'types', {'.mat'});

    results_dir = fullfile(basedir, 'cachedir', 'evaluation', ['p3d_' eval_split]);
    result_files = getFileNamesFromDirectory(results_dir, 'types', {'.mat'});
    valid_inds = ismember(result_files, csdm_files);
    result_files = result_files(valid_inds);

    gt_verts = load_p3d_meshes(p3d_dir, voc_class);
    gt_voxels_dir = fullfile(basedir, 'cachedir', 'p3d_eval', 'modelVoxels', voc_class);
    gt_voxels = load_p3d_voxels(gt_voxels_dir, length(gt_verts));

    %% Compute optimal transformation between predicted and gt frames
    anno_dir = fullfile(p3d_dir, 'Annotations', [voc_class '_pascal']);
    n_test = length(result_files);
    translations = zeros(n_test, 3);

    scales = zeros(n_test, 1);
    vertices_pred = {};
    subtypes = {};

    for ix = 1:length(result_files)
        var_pred = load(fullfile(results_dir, result_files{ix}));
        if ~exist('mesh_faces', 'var')
            mesh_faces = var_pred.faces;
        end
        vertices = var_pred.verts;
        vertices = vertices(:,[2 1 3]);
        vertices(:,1) = -vertices(:,1);
        vertices_pred{ix} = vertices;

        voc_id = strsplit(result_files{ix}(1:end-4), '_');
        voc_img_id = [voc_id{1} '_' voc_id{2}];

        rec_id = str2num(voc_id{3});
        var_gt = load(fullfile(anno_dir, voc_img_id));
        subtype = var_gt.record.objects(rec_id).cad_index;
        subtypes{ix} = subtype;

        [trans,scale] = estimateTransform(gt_verts{subtype}, vertices);
        translations(ix,:) = trans;
        scales(ix,:) = scale;
    end
    scale = mean(scales);
    trans = mean(translations);
    %scale = 0.5;
    %trans = [0, 0, 0];

    ious = zeros(n_test, 1);
    
    disp(trans);
    disp(scale);

    %% Compute IoUs
    for ix = 1:length(result_files)
        vertices = vertices_pred{ix};
        vertices = bsxfun(@plus, vertices, trans)*scale;
        FV = struct();
        FV.faces = mesh_faces;
        FV.vertices = (grid_size)*(vertices+0.5) + 0.5;

        pred_volume = polygon2voxel(FV, grid_size, 'none', false);
        gt_mesh = gt_verts{subtypes{ix}};

        %scatter3(gt_mesh(:,1), gt_mesh(:,2), gt_mesh(:,3), 'r.'); axis equal; hold on;
        %scatter3(vertices(:,1), vertices(:,2), vertices(:,3), 'b.'); axis equal; hold on;
        %disp(compute_iou(pred_volume, gt_voxels{subtypes{ix}}, 0.5));

        %keyboard;
        %close all;
        ious(ix) = compute_iou(pred_volume, gt_voxels{subtypes{ix}}, 0.5);
        
        %figure(1); imagesc(squeeze(gt_voxels{subtypes{ix}}(16,:,:)));

        %figure(2); imagesc(squeeze(pred_volume(16,:,:)));
    end
    fprintf('%s: mean iou %.4g\n', exp_name, mean(ious));
    %keyboard;

end

function iou = compute_iou(pred, gt, thresh)
    pred = double(pred > thresh);
    gt = double(gt > 0.5);
    intersection = sum(pred(:).*gt(:));
    total = sum(pred(:)) + sum(gt(:));
    iou = intersection/(total-intersection);
end


function gt_verts = load_p3d_meshes(p3d_dir, voc_class)
    models_all = load(fullfile(p3d_dir, 'CAD', voc_class));
    models_all = getfield(models_all, voc_class);

    n_models = length(models_all);
    vertices_gt = {};
    for i = 1:n_models
        vertices = models_all(i).vertices;
        vertices = vertices(:,[2 1 3]);
        vertices(:,1) = -vertices(:,1);
        gt_verts{i} = vertices;
    end
end

function gt_voxels = load_p3d_voxels(gt_voxels_dir, n_models)
    gt_voxels = {};
    for i = 1:n_models
        var_voxels = load(fullfile(gt_voxels_dir, num2str(i)));
        gt_voxels{i} = var_voxels.Volume;
    end
end

function [trans, scale] =  estimateTransform(vGt, vPred)
    vGtMin = min(vGt,[],1);
    vGtMax = max(vGt,[],1);
    
    vPredMin = min(vPred,[],1);
    vPredMax = max(vPred,[],1);
    
    trans = (vGtMin+vGtMax)/2 - (vPredMax + vPredMin)/2;
    scale = prod(vGtMax-vGtMin)/prod(vPredMax-vPredMin); scale = scale ^ (1/3);
end
