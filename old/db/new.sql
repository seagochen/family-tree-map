-- ******************************************
--   Postgresのsuperuserで実行
-- ******************************************

-- 管理ユーザー
--   PLACEHODLER: role name, password
CREATE ROLE fpa_admin WITH LOGIN CREATEROLE;
\echo 'Input the password of DB admin:'
\password fpa_admin

-- DB
--   PLACEHODLER: db name, owner name
CREATE DATABASE fpa_videos OWNER fpa_admin LC_COLLATE 'ja_JP.UTF-8' LC_CTYPE 'ja_JP.UTF-8' TEMPLATE template0;

\c fpa_videos


-------------------------------------------
-- Exclusion制約にoperator<> (Superuserの権限が必要)
CREATE EXTENSION IF NOT EXISTS btree_gist;
