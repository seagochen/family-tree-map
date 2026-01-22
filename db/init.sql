-- ******************************************
-- **** 本DBの管理ユーザーで実行
-- ******************************************

-------------------------------------------
-- PUBLICのデフォルトの関数実行権限を回収
--
ALTER DEFAULT PRIVILEGES FOR ROLE SESSION_USER REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC;


-------------------------------------------
-- 表を作成
--

-- 設定
--   Note: Use "numeric" for float numbers to make exact matching possible.
--   If both current value and default value are NULL, it means that the item
--     has no meaningful default value and the current value will be set later.
CREATE TABLE configurations (
    -- 設定項目名
    item_name varchar(64),
    -- 設定値
    current_value text DEFAULT NULL,
    -- 設定値のデフォルト値
    default_value text,
    -- 説明(データ型、用途など)
    description text NOT NULL CHECK (description <> ''),

    PRIMARY KEY (item_name)

    -- TODO:
    -- -- 設定値の有効性のチェック
    -- --   設定値はNULLであれば、戻り値もNULLになるので、ここにNULLをチェックする
    -- --   必要がない。
    -- CHECK (
    --     check_config_value(item_name, current_value)
    --     AND check_config_value(item_name, default_value)
    -- )
);


CREATE TABLE videos (
    id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    -- Relative to the root path of the video data directory.
    --   Unique constraint is for skipping duplication when importing.
    folder_path text NOT NULL UNIQUE CHECK (
        folder_path <> ''
        AND 0 = position('\' IN folder_path)
        AND '/' <> substring(folder_path FROM length(folder_path))
    )
);


CREATE TYPE label AS ENUM ('non_fire', 'fire');

-- Store information of all extracted clips.
--   Timestamps of frames are implictly defined by "start_time + INDEX * INTERVAL"
CREATE TABLE clips (
    video_id integer NOT NULL REFERENCES videos,
    -- Unit: ms
    start_time integer NOT NULL CHECK (start_time >= 0),
    -- Which class the clip represents
    class label NOT NULL,

    PRIMARY KEY (video_id, start_time)
);


CREATE TYPE spatial_model_type AS ENUM ('yolov8');

CREATE TABLE spatial_models (
    id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    architecture spatial_model_type NOT NULL,
    -- Use an empty string to represent no variants for the architecture
    variant text NOT NULL,
    training_started timestamp with time zone NOT NULL,
    feature_size integer NOT NULL,
    description text
);


-- Note: Use "numeric" for float numbers to make exact matching possible.
CREATE TABLE extraction_parameters (
    id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    spatial_model integer NOT NULL REFERENCES spatial_models,
    detection_threshold numeric NOT NULL CHECK (
        detection_threshold BETWEEN 0 AND 1
    ),
    fn_area_threshold numeric NOT NULL CHECK (
        fn_area_threshold <@ '(0,1]'::numrange
    ),
    tp_iou_threshold numeric NOT NULL CHECK (
        tp_iou_threshold <@ '(0,1]'::numrange
    ),

    UNIQUE (
        spatial_model, detection_threshold, fn_area_threshold, tp_iou_threshold
    )
);


-- Store all extracted feature vectors
CREATE TABLE features (
    video_id integer NOT NULL REFERENCES videos,
    -- Timestamp in the video (not clip) (unit: ms).
    time integer NOT NULL CHECK (time >= 0),
    extraction_param_id integer NOT NULL REFERENCES extraction_parameters,
    -- Which class the feature vector represents
    class label NOT NULL,
    vector real[] NOT NULL CHECK (cardinality(vector) > 0),

    PRIMARY KEY (video_id, time, extraction_param_id),

    -- Must have the same length if extraction parameters are the same
    EXCLUDE USING gist (extraction_param_id WITH =, cardinality(vector) WITH <>)
);


-- Note: Use "numeric" for float numbers to make exact matching possible.
-- Ratio of test set = 1 - training_set_ratio - validation_set_ratio
CREATE TABLE data_splits (
    id integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    time timestamp with time zone NOT NULL,
    extraction_param_id integer NOT NULL REFERENCES extraction_parameters,
    valid_feature_ratio numeric NOT NULL CHECK (
        valid_feature_ratio <@ '(0,1]'::numrange
    ),
    training_set_ratio numeric NOT NULL CHECK (
        training_set_ratio <@ '(0,1)'::numrange
    ),
    validation_set_ratio numeric NOT NULL CHECK (
        validation_set_ratio <@ '(0,1)'::numrange
    ),

    CHECK (training_set_ratio + validation_set_ratio <= 1)
);


CREATE TYPE dataset_type AS ENUM ('training', 'validation', 'testing');

CREATE TABLE datasets (
    split_id integer NOT NULL REFERENCES data_splits,
    category dataset_type NOT NULL,
    clip_video integer NOT NULL,
    clip_time integer NOT NULL,

    FOREIGN KEY (clip_video, clip_time) REFERENCES clips
);

-- For dataset split queries
CREATE INDEX dataset_type_index ON datasets (split_id, category);


-------------------------------------------
-- 関数を作成
--

-- TODO:
-- CREATE FUNCTION check_config_value(name text, value text) RETURNS boolean
-- AS $$
--     SELECT CASE item_name
--     END;
-- $$ LANGUAGE sql
--    IMMUTABLE
--    STRICT
--    PARALLEL SAFE;


-- Return:
--   Guarantees returning a non-NULL value to simplify application side code (no
--     need to do NULL check).
--   If the item doesn't exist or a NULL is retrieved, an exception will be
--     thrown.
CREATE FUNCTION get_config(item text) RETURNS text
AS $$
DECLARE
    value text;
BEGIN
    SELECT coalesce(current_value, default_value) INTO STRICT value
    FROM configurations WHERE item_name = item;

    IF value IS NULL THEN
        RAISE 'Configuration value not set: %', item;
    END IF;
    RETURN value;
END
$$ LANGUAGE plpgsql
   STABLE;

CREATE FUNCTION get_config_as_float(item text) RETURNS real
AS $$
    SELECT CAST(get_config(item) AS real);
$$ LANGUAGE sql
   STABLE;

CREATE FUNCTION get_config_as_numeric(item text) RETURNS numeric
AS $$
    SELECT CAST(get_config(item) AS numeric);
$$ LANGUAGE sql
   STABLE;

CREATE FUNCTION get_config_as_integer(item text) RETURNS integer
AS $$
    SELECT CAST(get_config(item) AS integer);
$$ LANGUAGE sql
   STABLE;

CREATE FUNCTION get_config_as_time(item text) RETURNS timestamp with time zone
AS $$
    SELECT CAST(get_config(item) AS timestamp with time zone);
$$ LANGUAGE sql
   STABLE;


-- Return: Model ID.
CREATE FUNCTION add_spatial_model(
    architecture spatial_model_type,
    variant text,
    training_started timestamp with time zone,
    feature_size integer,
    description text DEFAULT NULL
) RETURNS integer
AS $$
    INSERT INTO spatial_models
    VALUES (
        DEFAULT,
        architecture,
        variant,
        training_started,
        feature_size,
        description
    )
    RETURNING id;
$$ LANGUAGE sql;

-- If multiple models are matched, the one with the largest ID will be returned.
CREATE FUNCTION get_spatial_model_info(
    architecture spatial_model_type,
    variant text,
    training_started timestamp with time zone,
    OUT id integer,
    OUT feature_size integer
)
AS $$
    SELECT id, feature_size
    FROM spatial_models
    WHERE 
        architecture = architecture
        AND variant = variant
        AND training_started = training_started;
$$ LANGUAGE sql
   STABLE;


CREATE FUNCTION normalize_path(file_path text) RETURNS text
AS $$
BEGIN
    -- Trim leading and trailing whitespaces
    file_path := trim(E' \t' FROM file_path);
    -- "\" => "/"
    file_path := translate(file_path, '\', '/');
    -- Squeeze "/"'s
    file_path := regexp_replace(file_path, '/{2,}', '/', 'g');
    -- Remove hard link "." (e.g., "a/.", "./a", "a/./b")
    file_path := regexp_replace(file_path, '(?<=(^|/))\.(/|$)', '', 'g');
    -- Remove trailing "/" if there is one
    file_path := trim(TRAILING '/' FROM file_path);

    -- Verify
    IF 0 = char_length(file_path) THEN
        RAISE 'File path must not be empty after normalization: %', file_path;
    END IF;

    IF '/' = substring(file_path FOR 1) THEN
        RAISE
            'File path must not begin with "/" after normalization: %',
            file_path;
    END IF;

    IF file_path ~ '(?<=(^|/))\.\.(/|$)' THEN
        RAISE
            'File path must not contain hard link ".." after normalization: %',
            file_path;
    END IF;

    RETURN file_path;
END
$$ LANGUAGE plpgsql
   IMMUTABLE
   STRICT;

-- Description:
--   Skip if the video is already in the video table.
-- Return:
--   True if added, or false if skipped.
CREATE FUNCTION add_video(folder_path text) RETURNS boolean
AS $$
    WITH added AS (
        INSERT INTO videos
        VALUES (DEFAULT, normalize_path(folder_path))
        ON CONFLICT (folder_path) DO NOTHING
        RETURNING *
    )
    SELECT EXISTS(SELECT * FROM added);
$$ LANGUAGE sql;


-- Description:
--   Skip if the clip is already in the clip table.
-- Return:
--   True if added, or false if skipped.
CREATE FUNCTION add_clip(
    video_path text, start_time integer, class label
) RETURNS boolean
AS $$
DECLARE
    video_id integer;
BEGIN
    -- Raise an exception if the video is not found
    SELECT id INTO STRICT video_id
    FROM videos
    WHERE folder_path = normalize_path(video_path);

    INSERT INTO clips
    VALUES (video_id, start_time, class)
    ON CONFLICT DO NOTHING;

    RETURN FOUND;
END
$$ LANGUAGE plpgsql;

CREATE FUNCTION get_pending_clips(
    extraction_parameter_id integer
) RETURNS TABLE(
    video_id integer, video_path text, start_time integer
) AS $$
    -- For simplicity, check only the first frame of a clip
    SELECT c.video_id, v.folder_path, c.start_time
    FROM clips c
    JOIN videos v ON v.id = c.video_id
    WHERE NOT EXISTS(
        SELECT *
        FROM features f
        WHERE
            f.video_id = v.id AND f.time = c.start_time
            AND f.extraction_param_id = extraction_parameter_id
    )
$$ LANGUAGE sql;

-- Return: ID of the splitting.
CREATE FUNCTION split_clips(
    extraction_parameter_id integer,
    temporal_valid_feature_ratio numeric,
    training_set_ratio numeric,
    validation_set_ratio numeric
) RETURNS integer
AS $$
DECLARE
    split_id integer;
    frame_interval integer;
    clip_frame_num integer;
    clip_duration integer;
    num_valid_frames integer;
BEGIN
    -- Search split (pick the latest split if there are multiple ones)
    SELECT id INTO split_id
    FROM data_splits ds
    WHERE 
        ds.extraction_param_id = extraction_parameter_id
        AND ds.valid_feature_ratio = temporal_valid_feature_ratio
        AND ds.training_set_ratio = split_clips.training_set_ratio
        AND ds.validation_set_ratio = split_clips.validation_set_ratio
    ORDER BY time DESC;

    IF NOT FOUND THEN
        -- Generate split ID
        INSERT INTO data_splits (
            time,
            extraction_param_id,
            valid_feature_ratio,
            training_set_ratio,
            validation_set_ratio
        )
        VALUES (
            CURRENT_TIMESTAMP,
            extraction_parameter_id,
            temporal_valid_feature_ratio,
            training_set_ratio,
            validation_set_ratio
        )
        RETURNING id INTO split_id;

        SELECT get_config_as_integer('frame_sampling_interval')
        INTO STRICT frame_interval;
        SELECT get_config_as_integer('clip_frame_count')
        INTO STRICT clip_frame_num;
        clip_duration := frame_interval * clip_frame_num;
        num_valid_frames := round(
            clip_frame_num * temporal_valid_feature_ratio
        );

        -- Pick up only valid clips, split them, and write dataset table:
        --   Too many invalid feature vectors would make the clip unsuitable for
        --     representing its label. This approach is inspired by how we
        --     handle the case where fire being detected is temporarily blocked
        --     by some object (e.g., a flying bird). As long as we collect
        --     valid features from most of the fire-visible frames, invalid
        --     features contributed by the fire-invisible frames shouldn't
        --     affect the prediction result too much.
        INSERT INTO datasets
        SELECT
            split_id,
            assign_dataset(training_set_ratio, validation_set_ratio),
            c.video_id,
            c.start_time
        FROM clips c
        JOIN features f ON
            f.video_id = c.video_id AND f.class = c.class
            AND f.time <@ int4range(c.start_time, c.start_time + clip_duration)
        WHERE f.extraction_param_id = extraction_parameter_id
        GROUP BY c.video_id, c.start_time HAVING count(*) >= num_valid_frames;
    END IF;

    RETURN split_id;
END
$$ LANGUAGE plpgsql;

-- Return:
--   Clip data of the dataset.
--   The data will be shuffled before being returned.
CREATE FUNCTION get_dataset(
    split_id integer, category dataset_type
) RETURNS TABLE(features real[][], class label)
AS $$
    SELECT array_agg(f.vector ORDER BY f.time), c.class
    FROM datasets d
    JOIN data_splits s ON s.id = d.split_id
    JOIN clips c ON c.video_id = d.clip_video AND c.start_time = d.clip_time
    JOIN features f ON f.video_id = d.clip_video AND f.time <@ int4range(
        d.clip_time,
        d.clip_time + get_config_as_integer('frame_sampling_interval')
            * get_config_as_integer('clip_frame_count')
    )
    WHERE d.split_id = split_id AND d.category = category
        AND f.extraction_param_id = s.extraction_param_id
    GROUP BY d.clip_video, d.clip_time, c.class
    ORDER BY random();
$$ LANGUAGE sql;


CREATE FUNCTION assign_dataset(
    training_set_ratio real, validation_set_ratio real
) RETURNS dataset_type
AS $$
DECLARE
    pos real;
BEGIN
    pos := random();
    RETURN CASE
        WHEN pos <= training_set_ratio THEN 'training'
        WHEN pos <= training_set_ratio + validation_set_ratio THEN 'validation'
        ELSE 'testing'
    END;
END
$$ LANGUAGE plpgsql;


-- Return: ID of the set of parameters.
CREATE FUNCTION register_extraction_parameters(
    model integer,
    detection_threshold numeric,
    fn_area_threshold numeric,
    tp_iou_threshold numeric
) RETURNS integer
AS $$
    WITH inserted AS (
        INSERT INTO extraction_parameters
        VALUES (
            DEFAULT,
            model,
            detection_threshold,
            fn_area_threshold,
            tp_iou_threshold
        )
        ON CONFLICT DO NOTHING
        RETURNING id
    )
    SELECT id
    FROM inserted
    UNION
    SELECT id
    FROM extraction_parameters ep
    WHERE
        ep.spatial_model = model
        AND ep.detection_threshold = detection_threshold
        AND ep.fn_area_threshold = fn_area_threshold
        AND ep.tp_iou_threshold = tp_iou_threshold
    LIMIT 1;
$$ LANGUAGE sql;


-- Return:
--   True if inserted, or false if updated.
CREATE FUNCTION save_frame_feature(
    video_id integer,
    frame_time integer,
    extraction_parameter_id integer,
    vector real[],
    class label
) RETURNS boolean AS $$
    INSERT INTO features
    VALUES (video_id, frame_time, extraction_parameter_id, class, vector)
    ON CONFLICT (video_id, time, extraction_param_id) DO UPDATE
        SET class = EXCLUDED.class, vector = EXCLUDED.vector
    RETURNING xmax = 0;
$$ LANGUAGE sql;


-------------------------------------------
-- Configure
--
INSERT INTO configurations (item_name, default_value, description)
VALUES (
    'frame_sampling_interval',
    '200',
    'Frames must be sampled at a timestamp that is a multiple of the interval '
        '(integer | unit: ms | range: [1, ∞)).'
), (
    'clip_frame_count',
    '10',
    'The number of frames that a clip has (integer | range: [1, ∞)).'
), (
    'extractor_model_architecture',
    NULL,
    'The architecture of the spatial model (spatial_model_type).'
), (
    'extractor_model_variant',
    NULL,
    'The variant of the spatial model (text).'
), (
    'extractor_model_timestamp',
    NULL,
    'When the spatial model was created (timestamp with time zone).'
), (
    'extractor_detection_threshold',
    '0.25',
    'The threshold that the spatial model uses to detect fires '
        '(numeric | range: [0.0, 1.0]).'
), (
    'extractor_fn_area_threshold',
    '0.04', -- W >= 20% and H >= 20%
    'If the spatial model found nothing in a frame, the feature vector of the '
        'frame is labeled as "fire" only if the area of annotated fire is not '
        'less than the multiplication of this value and the area of the frame '
        '(numeric | range: [0.0, ∞)).'
), (
    'extractor_tp_iou_threshold',
    '0.5',
    'If the spatial model found fires in a frame, the feature vector of fire is '
        'labeled as "fire" only if the IoU of predicted boxes and annotated '
        'boxes is not less than this value (numeric | range: [0.0, ∞)).'
), (
    'temporal_valid_feature_ratio',
    '0.7',
    'The min ratio of valid feature vectors to make a clip a valid training '
        'sample. A feature vector is valid only if its label matches the label '
        'of the clip. A clip is valid only if the number of valid feature '
        'vectors is not less than the multiplication of the number of frames in '
        'the clip and this value. (numeric | range: [0.0, 1.0]).'
), (
    'dataset_training_ratio',
    '0.8',
    'The ratio of training data of the temporal model '
        '(numeric | range: (0.0, 1.0)).'
), (
    'dataset_validation_ratio',
    '0.12',
    'The ratio of validation data of the temporal model '
        '(numeric | range: (0.0, 1.0)).'
), (
    'training_batch_size',
    NULL,
    'Batch size used to train the temporal model (integer | range: [1, ∞)).'
), (
    'training_learning_rate',
    '0.001',
    'Learning rate used to train the temporal model '
        '(numeric | range: (0.0, ∞)).'
), (
    'training_weight_decay',
    '0.0004',
    'Weight decay used to train the temporal model (numeric | range: (0.0, ∞)).'
), (
    'training_dropout',
    '0.5',
    'Dropout used to train the temporal model (numeric | range: (0.0, ∞)).'
), (
    'training_number_of_epoches',
    '200',
    'How many epoches the temporal model will be trained '
        '(integer | range: [1, ∞)).'
);
