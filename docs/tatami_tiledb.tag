<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.9.5">
  <compound kind="file">
    <name>make_TileDbMatrix.hpp</name>
    <path>/github/workspace/include/tatami_tiledb/</path>
    <filename>make__TileDbMatrix_8hpp.html</filename>
    <namespace>tatami_tiledb</namespace>
    <member kind="function">
      <type>std::shared_ptr&lt; tatami::Matrix&lt; Value_, Index_ &gt; &gt;</type>
      <name>make_TileDbMatrix</name>
      <anchorfile>namespacetatami__tiledb.html</anchorfile>
      <anchor>a956c942c27f3c33e0a134d0b98a868c8</anchor>
      <arglist>(std::string uri, std::string attribute, const TileDbOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; tatami::Matrix&lt; Value_, Index_ &gt; &gt;</type>
      <name>make_TileDbMatrix</name>
      <anchorfile>namespacetatami__tiledb.html</anchorfile>
      <anchor>aec5503d305a2770e006d02b2fe3c6175</anchor>
      <arglist>(std::string uri, std::string attribute)</arglist>
    </member>
  </compound>
  <compound kind="file">
    <name>tatami_tiledb.hpp</name>
    <path>/github/workspace/include/tatami_tiledb/</path>
    <filename>tatami__tiledb_8hpp.html</filename>
    <namespace>tatami_tiledb</namespace>
  </compound>
  <compound kind="file">
    <name>TileDbDenseMatrix.hpp</name>
    <path>/github/workspace/include/tatami_tiledb/</path>
    <filename>TileDbDenseMatrix_8hpp.html</filename>
    <class kind="class">tatami_tiledb::TileDbDenseMatrix</class>
    <namespace>tatami_tiledb</namespace>
  </compound>
  <compound kind="file">
    <name>TileDbOptions.hpp</name>
    <path>/github/workspace/include/tatami_tiledb/</path>
    <filename>TileDbOptions_8hpp.html</filename>
    <class kind="struct">tatami_tiledb::TileDbOptions</class>
    <namespace>tatami_tiledb</namespace>
  </compound>
  <compound kind="file">
    <name>TileDbSparseMatrix.hpp</name>
    <path>/github/workspace/include/tatami_tiledb/</path>
    <filename>TileDbSparseMatrix_8hpp.html</filename>
    <class kind="class">tatami_tiledb::TileDbSparseMatrix</class>
    <namespace>tatami_tiledb</namespace>
  </compound>
  <compound kind="class">
    <name>tatami_tiledb::TileDbDenseMatrix</name>
    <filename>classtatami__tiledb_1_1TileDbDenseMatrix.html</filename>
    <templarg>typename Value_</templarg>
    <templarg>typename Index_</templarg>
    <templarg>bool transpose_</templarg>
    <base>VirtualDenseMatrix&lt; Value_, Index_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>TileDbDenseMatrix</name>
      <anchorfile>classtatami__tiledb_1_1TileDbDenseMatrix.html</anchorfile>
      <anchor>a6490c9a8d2a4740ba27056e9b93accf0</anchor>
      <arglist>(std::string uri, std::string attribute, const TileDbOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>TileDbDenseMatrix</name>
      <anchorfile>classtatami__tiledb_1_1TileDbDenseMatrix.html</anchorfile>
      <anchor>a320af27404e12340609d394b7324be39</anchor>
      <arglist>(std::string uri, std::string attribute)</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>tatami_tiledb::TileDbOptions</name>
    <filename>structtatami__tiledb_1_1TileDbOptions.html</filename>
    <member kind="variable">
      <type>size_t</type>
      <name>maximum_cache_size</name>
      <anchorfile>structtatami__tiledb_1_1TileDbOptions.html</anchorfile>
      <anchor>a318b648466b3dd661475bd672fbc1211</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>require_minimum_cache</name>
      <anchorfile>structtatami__tiledb_1_1TileDbOptions.html</anchorfile>
      <anchor>a8fec7a4cd15cace444af6c3fb3c9f7da</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>tatami_tiledb::TileDbSparseMatrix</name>
    <filename>classtatami__tiledb_1_1TileDbSparseMatrix.html</filename>
    <templarg>typename Value_</templarg>
    <templarg>typename Index_</templarg>
    <templarg>bool transpose_</templarg>
    <base>Matrix&lt; Value_, Index_ &gt;</base>
    <member kind="function">
      <type></type>
      <name>TileDbSparseMatrix</name>
      <anchorfile>classtatami__tiledb_1_1TileDbSparseMatrix.html</anchorfile>
      <anchor>ae876ac63c20f62517e79b24c30e05a65</anchor>
      <arglist>(std::string uri, std::string attribute, const TileDbOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type></type>
      <name>TileDbSparseMatrix</name>
      <anchorfile>classtatami__tiledb_1_1TileDbSparseMatrix.html</anchorfile>
      <anchor>a31faad0efecbec021fade3e97506ff91</anchor>
      <arglist>(std::string uri, std::string attribute)</arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>tatami_tiledb</name>
    <filename>namespacetatami__tiledb.html</filename>
    <class kind="class">tatami_tiledb::TileDbDenseMatrix</class>
    <class kind="struct">tatami_tiledb::TileDbOptions</class>
    <class kind="class">tatami_tiledb::TileDbSparseMatrix</class>
    <member kind="function">
      <type>std::shared_ptr&lt; tatami::Matrix&lt; Value_, Index_ &gt; &gt;</type>
      <name>make_TileDbMatrix</name>
      <anchorfile>namespacetatami__tiledb.html</anchorfile>
      <anchor>a956c942c27f3c33e0a134d0b98a868c8</anchor>
      <arglist>(std::string uri, std::string attribute, const TileDbOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>std::shared_ptr&lt; tatami::Matrix&lt; Value_, Index_ &gt; &gt;</type>
      <name>make_TileDbMatrix</name>
      <anchorfile>namespacetatami__tiledb.html</anchorfile>
      <anchor>aec5503d305a2770e006d02b2fe3c6175</anchor>
      <arglist>(std::string uri, std::string attribute)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>tatami for TileDB matrices</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__github_workspace_README</docanchor>
  </compound>
</tagfile>
