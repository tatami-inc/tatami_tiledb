<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.9.8">
  <compound kind="file">
    <name>DenseMatrix.hpp</name>
    <path>tatami_tiledb/</path>
    <filename>DenseMatrix_8hpp.html</filename>
    <class kind="struct">tatami_tiledb::DenseMatrixOptions</class>
    <class kind="class">tatami_tiledb::DenseMatrix</class>
    <namespace>tatami_tiledb</namespace>
  </compound>
  <compound kind="file">
    <name>serialize.hpp</name>
    <path>tatami_tiledb/</path>
    <filename>serialize_8hpp.html</filename>
    <namespace>tatami_tiledb</namespace>
  </compound>
  <compound kind="file">
    <name>SparseMatrix.hpp</name>
    <path>tatami_tiledb/</path>
    <filename>SparseMatrix_8hpp.html</filename>
    <class kind="struct">tatami_tiledb::SparseMatrixOptions</class>
    <class kind="class">tatami_tiledb::SparseMatrix</class>
    <namespace>tatami_tiledb</namespace>
  </compound>
  <compound kind="file">
    <name>tatami_tiledb.hpp</name>
    <path>tatami_tiledb/</path>
    <filename>tatami__tiledb_8hpp.html</filename>
    <namespace>tatami_tiledb</namespace>
  </compound>
  <compound kind="class">
    <name>tatami_tiledb::DenseMatrix</name>
    <filename>classtatami__tiledb_1_1DenseMatrix.html</filename>
    <templarg>typename Value_</templarg>
    <templarg>typename Index_</templarg>
    <base>Matrix&lt; Value_, Index_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>DenseMatrix</name>
      <anchorfile>classtatami__tiledb_1_1DenseMatrix.html</anchorfile>
      <anchor>a45797084654f5dff590d3d7feeb17ad4</anchor>
      <arglist>(const std::string &amp;uri, std::string attribute, tiledb::Context ctx, const DenseMatrixOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DenseMatrix</name>
      <anchorfile>classtatami__tiledb_1_1DenseMatrix.html</anchorfile>
      <anchor>a714b717e3c234454a12bd18e5a7c1d3c</anchor>
      <arglist>(const std::string &amp;uri, std::string attribute, const DenseMatrixOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>DenseMatrix</name>
      <anchorfile>classtatami__tiledb_1_1DenseMatrix.html</anchorfile>
      <anchor>ad3a8e80f5bc695fe4938ec41d5f9b45a</anchor>
      <arglist>(const std::string &amp;uri, std::string attribute)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>tatami_tiledb::DenseMatrixOptions</name>
    <filename>structtatami__tiledb_1_1DenseMatrixOptions.html</filename>
    <member kind="variable">
      <type>size_t</type>
      <name>maximum_cache_size</name>
      <anchorfile>structtatami__tiledb_1_1DenseMatrixOptions.html</anchorfile>
      <anchor>a48fa577abbde5514e1a35c3941fbe810</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>require_minimum_cache</name>
      <anchorfile>structtatami__tiledb_1_1DenseMatrixOptions.html</anchorfile>
      <anchor>a019938987bd2dfd3a9c9e5137678c504</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>tatami_tiledb::SparseMatrix</name>
    <filename>classtatami__tiledb_1_1SparseMatrix.html</filename>
    <templarg>typename Value_</templarg>
    <templarg>typename Index_</templarg>
    <base>Matrix&lt; Value_, Index_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>SparseMatrix</name>
      <anchorfile>classtatami__tiledb_1_1SparseMatrix.html</anchorfile>
      <anchor>afbe309f2fe08bd91cc5b9188e88a73ec</anchor>
      <arglist>(const std::string &amp;uri, std::string attribute, tiledb::Context ctx, const SparseMatrixOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>SparseMatrix</name>
      <anchorfile>classtatami__tiledb_1_1SparseMatrix.html</anchorfile>
      <anchor>a5ee3846a5281cd9f8316c0cb60fa1855</anchor>
      <arglist>(const std::string &amp;uri, std::string attribute, const SparseMatrixOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>SparseMatrix</name>
      <anchorfile>classtatami__tiledb_1_1SparseMatrix.html</anchorfile>
      <anchor>ab0b5f4f04116d20e26f4d44e0260909e</anchor>
      <arglist>(const std::string &amp;uri, std::string attribute)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>tatami_tiledb::SparseMatrixOptions</name>
    <filename>structtatami__tiledb_1_1SparseMatrixOptions.html</filename>
    <member kind="variable">
      <type>size_t</type>
      <name>maximum_cache_size</name>
      <anchorfile>structtatami__tiledb_1_1SparseMatrixOptions.html</anchorfile>
      <anchor>a7c809c3c948980fb7b7216bb73180a9f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>require_minimum_cache</name>
      <anchorfile>structtatami__tiledb_1_1SparseMatrixOptions.html</anchorfile>
      <anchor>aadbd63a4c848f631692ffc5f4ef75578</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>tatami_tiledb</name>
    <filename>namespacetatami__tiledb.html</filename>
    <class kind="class">tatami_tiledb::DenseMatrix</class>
    <class kind="struct">tatami_tiledb::DenseMatrixOptions</class>
    <class kind="class">tatami_tiledb::SparseMatrix</class>
    <class kind="struct">tatami_tiledb::SparseMatrixOptions</class>
    <member kind="function">
      <type>void</type>
      <name>serialize</name>
      <anchorfile>namespacetatami__tiledb.html</anchorfile>
      <anchor>a63671723fdbc1325860b48d9b629dd2c</anchor>
      <arglist>(Function_ fun)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>tatami for TileDB matrices</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
