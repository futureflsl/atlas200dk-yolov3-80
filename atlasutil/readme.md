
<!DOCTYPE html>

<html>
  <head>
    <meta charset="utf-8" />
    <title>Welcome to HuaweiAtlas’s documentation! &#8212; atlasutil [1.0] documentation</title>
  <meta name="viewport" content="width=device-width, initial-scale=0.9, maximum-scale=0.9" />

  </head><body>
            
  <div class="section" id="welcome-to-test-s-documentation">
<h1>Welcome to HuaweiAtlas’s documentation!<a class="headerlink" href="#welcome-to-test-s-documentation" title="Permalink to this headline">¶</a></h1>
<div class="toctree-wrapper compound">
</div>
</div>
<div class="section" id="module-atlasutil.ai.graph">
<span id="api"></span><h1>API<a class="headerlink" href="#module-atlasutil.ai.graph" title="Permalink to this headline">¶</a></h1>
<dl class="py class">
<dt id="atlasutil.ai.graph.Graph">
<em class="property">class </em><code class="sig-prename descclassname">atlasutil.ai.graph.</code><code class="sig-name descname">Graph</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">model_path</span></em><span class="sig-paren">)</span><a class="headerlink" href="#atlasutil.ai.graph.Graph" title="Permalink to this definition">¶</a></dt>
<dd><p>Graph类，对hiai库中graph行为进行封装，包括创建graph、推理、destroy graph等</p>
<dl class="py method">
<dt id="atlasutil.ai.graph.Graph.CreateGraph">
<code class="sig-name descname">CreateGraph</code><span class="sig-paren">(</span><span class="sig-paren">)</span><a class="headerlink" href="#atlasutil.ai.graph.Graph.CreateGraph" title="Permalink to this definition">¶</a></dt>
<dd><p>创建graph</p>
<p>无需用户显示调用，已集成在初始化方法中</p>
<dl class="simple">
<dt>Returns:</dt><dd><p>graph: 返回创建的graph</p>
</dd>
</dl>
</dd></dl>

<dl class="py method">
<dt id="atlasutil.ai.graph.Graph.CreateNntensorList">
<code class="sig-name descname">CreateNntensorList</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input_data</span></em><span class="sig-paren">)</span><a class="headerlink" href="#atlasutil.ai.graph.Graph.CreateNntensorList" title="Permalink to this definition">¶</a></dt>
<dd><p>创建NNTensorList</p>
<p>创建NNTensorList，内部接口，供推理接口调用，对输入数据进行封装</p>
<dl class="simple">
<dt>Args:</dt><dd><p>input_data (numpy.ndarray): 待推理的数据</p>
</dd>
<dt>Returns:</dt><dd><p>nntensorList (hiai.NNTensorList): NNTensorList对象</p>
</dd>
</dl>
</dd></dl>

<dl class="py method">
<dt id="atlasutil.ai.graph.Graph.Inference">
<code class="sig-name descname">Inference</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input_data</span></em><span class="sig-paren">)</span><a class="headerlink" href="#atlasutil.ai.graph.Graph.Inference" title="Permalink to this definition">¶</a></dt>
<dd><p>推理接口</p>
<p>输入待推理的数据，把数据转为NNTensorList并进行推理</p>
<dl class="simple">
<dt>Args:</dt><dd><p>input_data (numpy.ndarray): 待推理的数据</p>
</dd>
<dt>Returns:</dt><dd><p>list, 包含推理结果的list，list中为模型的输出tensor</p>
</dd>
</dl>
</dd></dl>

<dl class="py method">
<dt id="atlasutil.ai.graph.Graph.create_graph_with_dvpp">
<code class="sig-name descname">create_graph_with_dvpp</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">resize_cfg</span></em><span class="sig-paren">)</span><a class="headerlink" href="#atlasutil.ai.graph.Graph.create_graph_with_dvpp" title="Permalink to this definition">¶</a></dt>
<dd><p>创建graph，并未graph配置crop和reszie算子（用dvpp，效率较高）</p>
<p>接口测试中，不建议使用</p>
<dl class="simple">
<dt>Args:</dt><dd><p>resize_cfg (tuple): resize的目标尺寸，(宽，高)</p>
</dd>
<dt>Returns:</dt><dd><p>graph: 返回创建的graph</p>
</dd>
</dl>
</dd></dl>

</dd></dl>

<span class="target" id="module-atlasutil.ai.post_process"></span><dl class="py function">
<dt id="atlasutil.ai.post_process.FasterRCNNPostProcess">
<code class="sig-prename descclassname">atlasutil.ai.post_process.</code><code class="sig-name descname">FasterRCNNPostProcess</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">resultList</span></em>, <em class="sig-param"><span class="n">confidence_threshold</span></em><span class="sig-paren">)</span><a class="headerlink" href="#atlasutil.ai.post_process.FasterRCNNPostProcess" title="Permalink to this definition">¶</a></dt>
<dd><p>faster rcnn后处理接口</p>
<p>faster rcnn后处理接口，未使用，待验证</p>
<dl class="simple">
<dt>Args:</dt><dd><p>resultList (numpy.ndarray) : 模型推理结果中的numpy数组
confidence_threshold (float) : 置信度，用于过滤低于该置信度的检测结果</p>
</dd>
<dt>Returns:</dt><dd><p>result_bbox (list) : 筛选后的检测结果的列表，聊表中每个元素为一个长度为6的列表，依次为：两个坐标点坐标，类别，置信度</p>
</dd>
</dl>
</dd></dl>

<dl class="py function">
<dt id="atlasutil.ai.post_process.GenerateTopNClassifyResult">
<code class="sig-prename descclassname">atlasutil.ai.post_process.</code><code class="sig-name descname">GenerateTopNClassifyResult</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">resultList</span></em>, <em class="sig-param"><span class="n">n</span></em><span class="sig-paren">)</span><a class="headerlink" href="#atlasutil.ai.post_process.GenerateTopNClassifyResult" title="Permalink to this definition">¶</a></dt>
<dd><p>分类模型后处理接口</p>
<p>分类模型后处理接口，输入推理结果和n，返回前n个置信度和对应的类别标签</p>
<dl class="simple">
<dt>Args:</dt><dd><p>resultList (list) : 推理结果，graph.Inference接口返回值
n (int) : 对置信度从高到低排序后要获取的结果的数量</p>
</dd>
<dt>Returns:</dt><dd><p>topNArray: 前n个置信度（从高到低排序）
confidenceIndex: 前n个置信度（从高到低排序）对应的index</p>
</dd>
</dl>
</dd></dl>

<dl class="py function">
<dt id="atlasutil.ai.post_process.SSDPostProcess">
<code class="sig-prename descclassname">atlasutil.ai.post_process.</code><code class="sig-name descname">SSDPostProcess</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">inference_result</span></em>, <em class="sig-param"><span class="n">image_resolution</span></em>, <em class="sig-param"><span class="n">confidence_threshold</span></em>, <em class="sig-param"><span class="n">labels</span><span class="o">=</span><span class="default_value">[]</span></em><span class="sig-paren">)</span><a class="headerlink" href="#atlasutil.ai.post_process.SSDPostProcess" title="Permalink to this definition">¶</a></dt>
<dd><p>SSD后处理接口</p>
<p>SSD后处理接口</p>
<dl class="simple">
<dt>Args:</dt><dd><p>resultList (list) : 推理结果，graph.Inference接口返回值
image_resolution (tuple) : 图像分辨率，(高，宽)，可以通过numpy.ndarray.shape获得
confidence_threshold (float) : 置信度，用于过滤掉低于该置信度的结果
labels: 模型能检测的类别标签</p>
</dd>
<dt>Returns:</dt><dd><p>detection_result_list (list) : 筛选出来的目标的信息(detection_item)的列表，包括：detection_item.attr目标类别，detection_item.confidence目标置信度，detection_item.lt.x目标框的左上x坐标，detection_item.lt.y目标框右上角y坐标，detection_item.rb.x目标框右下角x坐标，detection_item.rb.y目标框右下角y坐标，detection_item.result_text如果输入了标签，还会返回类别+对应的精确到百分位的置信度</p>
</dd>
</dl>
</dd></dl>

<dl class="py function">
<dt id="atlasutil.ai.post_process.Yolov3_post_process">
<code class="sig-prename descclassname">atlasutil.ai.post_process.</code><code class="sig-name descname">Yolov3_post_process</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">resultList</span></em>, <em class="sig-param"><span class="n">confidence_threshold</span></em>, <em class="sig-param"><span class="n">nms_threshold</span></em>, <em class="sig-param"><span class="n">model_shape</span></em>, <em class="sig-param"><span class="n">img_shape</span></em>, <em class="sig-param"><span class="n">labels</span><span class="o">=</span><span class="default_value">[]</span></em>, <em class="sig-param"><span class="n">anchors</span><span class="o">=</span><span class="default_value">[[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]</span></em><span class="sig-paren">)</span><a class="headerlink" href="#atlasutil.ai.post_process.Yolov3_post_process" title="Permalink to this definition">¶</a></dt>
<dd><p>Yolov3后处理接口</p>
<p>Yolov3后处理接口</p>
<dl class="simple">
<dt>Args:</dt><dd><p>resultList (list) : 推理结果，graph.Inference接口返回值
confidence_threshold (float) : 置信度，用于过滤掉低于该置信度的结果
nms_threshold (float) : NMS阈值
model_shape (tuple) : 模型输入的分辨率，（高，宽）
img_shape (tuple) : 输入图像的原始分辨率，（高，宽）
labels (list) : 模型能检测的标签类别
anchors: anchor box参数</p>
</dd>
<dt>Returns:</dt><dd><p>detection_result_list (list) : 筛选出来的目标的信息(detection_item)的列表，包括：detection_item.attr目标类别，detection_item.confidence目标置信度，detection_item.lt.x目标框的左上x坐标，detection_item.lt.y目标框右上角y坐标，detection_item.rb.x目标框右下角x坐标，detection_item.rb.y目标框右下角y坐标，detection_item.result_text如果输入了标签，还会返回类别+对应的精确到百分位的置信度</p>
</dd>
</dl>
</dd></dl>

<span class="target" id="module-atlasutil.dvpp_process"></span></div>


          </div>
          
        </div>
      </div>
      <div class="sphinxsidebar" role="navigation" aria-label="main navigation">
        <div class="sphinxsidebarwrapper">
<h1 class="logo"><a href="#">atlasutil</a></h1>








        </div>
      </div>
      <div class="clearer"></div>
    </div>

    

    
  </body>
</html>