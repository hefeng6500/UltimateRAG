"""
Neo4j GraphRAG 演示脚本

演示如何使用 GraphRAG 功能，将知识图谱数据存储到 Neo4j 数据库。

使用前确保：
1. Neo4j 已启动（默认端口 7687）
2. 在项目根目录创建 .env 文件，配置以下变量：
   - NEO4J_URI=neo4j://127.0.0.1:7687
   - NEO4J_USERNAME=neo4j
   - NEO4J_PASSWORD=你的密码
   - NEO4J_DATABASE=neo4j
   - GRAPH_STORE_TYPE=neo4j
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from loguru import logger

from src.stage_4.config import Stage4Config, get_stage4_config
from src.stage_4.graph_rag import (
    Entity,
    EntityType,
    Relation,
    RelationType,
    KnowledgeGraph,
    Neo4jGraphStore,
    GraphRAGChain,
    create_graph_store,
)


def create_sample_knowledge_graph() -> KnowledgeGraph:
    """
    创建示例知识图谱
    
    Returns:
        KnowledgeGraph: 包含示例数据的知识图谱
    """
    kg = KnowledgeGraph()
    
    # ==================
    # 1. 添加实体
    # ==================
    entities = [
        # 人物
        Entity(
            name="雷军",
            type=EntityType.PERSON,
            description="小米集团创始人、董事长兼CEO",
            aliases=["Lei Jun"],
        ),
        Entity(
            name="马云",
            type=EntityType.PERSON,
            description="阿里巴巴集团创始人",
            aliases=["Jack Ma"],
        ),
        Entity(
            name="任正非",
            type=EntityType.PERSON,
            description="华为公司创始人、CEO",
            aliases=["Ren Zhengfei"],
        ),
        Entity(
            name="马化腾",
            type=EntityType.PERSON,
            description="腾讯公司创始人、董事会主席兼CEO",
            aliases=["Pony Ma"],
        ),
        
        # 公司/组织
        Entity(
            name="小米集团",
            type=EntityType.ORGANIZATION,
            description="中国科技公司，主营智能手机、智能硬件",
            aliases=["Xiaomi", "小米"],
        ),
        Entity(
            name="阿里巴巴",
            type=EntityType.ORGANIZATION,
            description="中国电商和科技巨头",
            aliases=["Alibaba", "阿里"],
        ),
        Entity(
            name="华为",
            type=EntityType.ORGANIZATION,
            description="中国通信和科技公司",
            aliases=["Huawei"],
        ),
        Entity(
            name="腾讯",
            type=EntityType.ORGANIZATION,
            description="中国互联网科技公司",
            aliases=["Tencent"],
        ),
        Entity(
            name="苹果公司",
            type=EntityType.ORGANIZATION,
            description="美国科技公司",
            aliases=["Apple"],
        ),
        
        # 地点
        Entity(
            name="北京",
            type=EntityType.LOCATION,
            description="中国首都",
            aliases=["Beijing"],
        ),
        Entity(
            name="深圳",
            type=EntityType.LOCATION,
            description="中国科技创新中心",
            aliases=["Shenzhen"],
        ),
        Entity(
            name="杭州",
            type=EntityType.LOCATION,
            description="中国电商之都",
            aliases=["Hangzhou"],
        ),
        
        # 产品
        Entity(
            name="小米手机",
            type=EntityType.PRODUCT,
            description="小米集团生产的智能手机",
            aliases=["Mi Phone", "Xiaomi Phone"],
        ),
        Entity(
            name="华为Mate系列",
            type=EntityType.PRODUCT,
            description="华为旗舰手机系列",
            aliases=["Huawei Mate"],
        ),
        Entity(
            name="微信",
            type=EntityType.PRODUCT,
            description="腾讯开发的即时通讯软件",
            aliases=["WeChat"],
        ),
        Entity(
            name="淘宝",
            type=EntityType.PRODUCT,
            description="阿里巴巴旗下电商平台",
            aliases=["Taobao"],
        ),
        
        # 概念
        Entity(
            name="人工智能",
            type=EntityType.CONCEPT,
            description="模拟人类智能的计算机技术",
            aliases=["AI", "Artificial Intelligence"],
        ),
        Entity(
            name="5G技术",
            type=EntityType.CONCEPT,
            description="第五代移动通信技术",
            aliases=["5G"],
        ),
        
        # 事件
        Entity(
            name="小米SU7发布会",
            type=EntityType.EVENT,
            description="小米首款汽车发布会",
            aliases=["Xiaomi SU7 Launch"],
        ),
    ]
    
    logger.info(f"📦 添加 {len(entities)} 个实体...")
    for entity in entities:
        kg.add_entity(entity)
    
    # ==================
    # 2. 添加关系
    # ==================
    relations = [
        # 创始人关系
        Relation(
            source="雷军",
            target="小米集团",
            relation_type=RelationType.FOUNDED,
            description="雷军于2010年创立小米",
        ),
        Relation(
            source="马云",
            target="阿里巴巴",
            relation_type=RelationType.FOUNDED,
            description="马云于1999年创立阿里巴巴",
        ),
        Relation(
            source="任正非",
            target="华为",
            relation_type=RelationType.FOUNDED,
            description="任正非于1987年创立华为",
        ),
        Relation(
            source="马化腾",
            target="腾讯",
            relation_type=RelationType.FOUNDED,
            description="马化腾于1998年创立腾讯",
        ),
        
        # 管理关系
        Relation(
            source="雷军",
            target="小米集团",
            relation_type=RelationType.MANAGES,
            description="雷军担任小米集团CEO",
        ),
        Relation(
            source="任正非",
            target="华为",
            relation_type=RelationType.MANAGES,
            description="任正非担任华为CEO",
        ),
        
        # 地点关系
        Relation(
            source="小米集团",
            target="北京",
            relation_type=RelationType.LOCATED_IN,
            description="小米集团总部位于北京",
        ),
        Relation(
            source="阿里巴巴",
            target="杭州",
            relation_type=RelationType.LOCATED_IN,
            description="阿里巴巴总部位于杭州",
        ),
        Relation(
            source="华为",
            target="深圳",
            relation_type=RelationType.LOCATED_IN,
            description="华为总部位于深圳",
        ),
        Relation(
            source="腾讯",
            target="深圳",
            relation_type=RelationType.LOCATED_IN,
            description="腾讯总部位于深圳",
        ),
        
        # 产品关系
        Relation(
            source="小米集团",
            target="小米手机",
            relation_type=RelationType.PRODUCES,
            description="小米集团生产小米手机",
        ),
        Relation(
            source="华为",
            target="华为Mate系列",
            relation_type=RelationType.PRODUCES,
            description="华为生产Mate系列手机",
        ),
        Relation(
            source="腾讯",
            target="微信",
            relation_type=RelationType.PRODUCES,
            description="腾讯开发微信",
        ),
        Relation(
            source="阿里巴巴",
            target="淘宝",
            relation_type=RelationType.PRODUCES,
            description="阿里巴巴运营淘宝",
        ),
        
        # 竞争关系
        Relation(
            source="小米集团",
            target="华为",
            relation_type=RelationType.COMPETES_WITH,
            description="小米与华为在手机市场竞争",
        ),
        Relation(
            source="小米集团",
            target="苹果公司",
            relation_type=RelationType.COMPETES_WITH,
            description="小米与苹果在智能手机市场竞争",
        ),
        Relation(
            source="华为",
            target="苹果公司",
            relation_type=RelationType.COMPETES_WITH,
            description="华为与苹果在全球手机市场竞争",
        ),
        
        # 技术相关
        Relation(
            source="华为",
            target="5G技术",
            relation_type=RelationType.RELATED_TO,
            description="华为是5G技术领先企业",
        ),
        Relation(
            source="小米集团",
            target="人工智能",
            relation_type=RelationType.RELATED_TO,
            description="小米在智能家居中应用AI技术",
        ),
        Relation(
            source="腾讯",
            target="人工智能",
            relation_type=RelationType.RELATED_TO,
            description="腾讯大力发展AI技术",
        ),
        Relation(
            source="阿里巴巴",
            target="人工智能",
            relation_type=RelationType.RELATED_TO,
            description="阿里巴巴云计算和AI技术",
        ),
        
        # 事件参与
        Relation(
            source="雷军",
            target="小米SU7发布会",
            relation_type=RelationType.PARTICIPATES_IN,
            description="雷军主持小米SU7发布会",
        ),
        Relation(
            source="小米集团",
            target="小米SU7发布会",
            relation_type=RelationType.PARTICIPATES_IN,
            description="小米举办SU7发布会",
        ),
    ]
    
    logger.info(f"🔗 添加 {len(relations)} 条关系...")
    for relation in relations:
        kg.add_relation(relation)
    
    return kg


def demo_neo4j_storage():
    """
    演示将知识图谱存储到 Neo4j
    """
    print("\n" + "=" * 60)
    print("🎯 Neo4j 图存储演示")
    print("=" * 60 + "\n")
    
    # 加载配置
    load_dotenv()
    
    # 创建 Neo4j 存储
    logger.info("📊 初始化 Neo4j 图存储...")
    
    try:
        store = Neo4jGraphStore()
    except Exception as e:
        logger.error(f"❌ 无法连接到 Neo4j: {e}")
        logger.info("请确保:")
        logger.info("  1. Neo4j 已启动")
        logger.info("  2. .env 文件中配置了正确的 NEO4J_PASSWORD")
        return
    
    # 创建示例知识图谱
    logger.info("\n🔨 创建示例知识图谱...")
    kg = create_sample_knowledge_graph()
    
    # 显示图谱统计
    stats = kg.get_statistics()
    print(f"\n📈 知识图谱统计:")
    print(f"   - 实体数量: {stats['num_nodes']}")
    print(f"   - 关系数量: {stats['num_edges']}")
    print(f"   - 实体类型: {stats.get('entity_type_counts', {})}")
    print(f"   - 关系类型: {stats.get('relation_type_counts', {})}")
    
    # 保存到 Neo4j
    logger.info("\n💾 保存知识图谱到 Neo4j...")
    store.save(kg, name="demo_graph")
    
    # 验证保存
    logger.info("\n🔍 验证保存结果...")
    loaded_kg = store.load("demo_graph")
    if loaded_kg:
        print(f"   ✅ 成功加载图谱:")
        print(f"      - 实体: {loaded_kg.num_nodes}")
        print(f"      - 关系: {loaded_kg.num_edges}")
    
    # 列出所有图谱
    graphs = store.list_graphs()
    print(f"\n📚 Neo4j 中的所有图谱: {graphs}")
    
    # 关闭连接
    store.close()
    
    print("\n" + "=" * 60)
    print("✅ 演示完成！")
    print("   现在可以在 Neo4j Browser 中查看知识图谱")
    print("   访问: http://localhost:7474")
    print("   查询示例: MATCH (n:Entity) RETURN n LIMIT 50")
    print("=" * 60)


def demo_graph_rag_with_neo4j():
    """
    演示使用 Neo4j 作为后端的 GraphRAG
    """
    print("\n" + "=" * 60)
    print("🎯 GraphRAG + Neo4j 演示")
    print("=" * 60 + "\n")
    
    # 加载配置
    load_dotenv()
    
    # 设置使用 Neo4j 存储
    os.environ["GRAPH_STORE_TYPE"] = "neo4j"
    
    # 重新加载配置
    from src.stage_4.config import Stage4Config
    config = Stage4Config.from_stage3_config()
    
    if config.graph_store_type != "neo4j":
        logger.warning("⚠️ GRAPH_STORE_TYPE 未设置为 neo4j")
    
    # 创建 GraphRAG 链
    logger.info("📊 初始化 GraphRAG (Neo4j 后端)...")
    
    try:
        # 创建 Neo4j 图存储
        store = create_graph_store(store_type="neo4j", config=config)
        
        # 检查是否已有图谱
        if store.exists("graphrag_demo"):
            logger.info("📂 发现已有图谱，加载中...")
            kg = store.load("graphrag_demo")
        else:
            logger.info("🔨 创建新的知识图谱...")
            kg = create_sample_knowledge_graph()
            store.save(kg, "graphrag_demo")
        
        # 显示统计
        stats = kg.get_statistics()
        print(f"\n📈 知识图谱统计:")
        print(f"   - 实体: {stats['num_nodes']}")
        print(f"   - 关系: {stats['num_edges']}")
        
        # 演示图谱查询
        print("\n" + "-" * 60)
        print("📊 图谱查询演示")
        print("-" * 60)
        
        # 查询实体
        entity = kg.get_entity_by_name("雷军")
        if entity:
            print(f"\n👤 实体查询: 雷军")
            print(f"   类型: {entity.type.value}")
            print(f"   描述: {entity.description}")
            
            # 查询邻居
            neighbors = kg.get_neighbors("雷军", hops=1)
            print(f"   直接关联: {len(neighbors.nodes)} 个实体, {len(neighbors.edges)} 条关系")
            for rel in neighbors.edges[:5]:
                print(f"     - {rel.source} --[{rel.relation_type.value}]--> {rel.target}")
        
        # 查找路径
        print(f"\n🛤️ 路径查询: 雷军 → 人工智能")
        path = kg.find_path("雷军", "人工智能")
        if path:
            for entity, relation in path:
                if relation:
                    print(f"   [{entity.name}] --{relation.relation_type.value}-->")
                else:
                    print(f"   [{entity.name}]")
        
        # 关闭存储
        store.close()
        
    except Exception as e:
        logger.error(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✅ GraphRAG + Neo4j 演示完成！")
    print("=" * 60)


def demo_interactive():
    """
    交互式演示
    """
    print("\n" + "=" * 60)
    print("🎯 交互式 Neo4j 知识图谱演示")
    print("=" * 60 + "\n")
    
    # 加载配置
    load_dotenv()
    
    # 创建 Neo4j 存储
    try:
        store = Neo4jGraphStore()
    except Exception as e:
        logger.error(f"❌ 无法连接到 Neo4j: {e}")
        return
    
    # 加载或创建图谱
    kg = store.load("interactive_demo")
    if kg is None:
        logger.info("创建新的知识图谱...")
        kg = create_sample_knowledge_graph()
        store.save(kg, "interactive_demo")
    else:
        logger.info(f"加载已有图谱: {kg.num_nodes} 实体, {kg.num_edges} 关系")
    
    print("\n" + "-" * 60)
    print("💬 交互式查询 (输入 'quit' 退出)")
    print("   命令:")
    print("   - entity <名称>  : 查询实体信息")
    print("   - path <起点> <终点> : 查找路径")
    print("   - neighbors <名称> : 查询邻居")
    print("   - stats : 显示统计信息")
    print("-" * 60)
    
    while True:
        try:
            cmd = input("\n>>> ").strip()
        except EOFError:
            break
        
        if not cmd:
            continue
        
        if cmd.lower() in ['quit', 'exit', 'q']:
            break
        
        parts = cmd.split()
        action = parts[0].lower()
        
        if action == "entity" and len(parts) >= 2:
            name = " ".join(parts[1:])
            entity = kg.get_entity_by_name(name)
            if entity:
                print(f"\n📦 实体: {entity.name}")
                print(f"   类型: {entity.type.value}")
                print(f"   描述: {entity.description}")
                print(f"   别名: {entity.aliases}")
            else:
                print(f"❌ 未找到实体: {name}")
        
        elif action == "path" and len(parts) >= 3:
            source = parts[1]
            target = parts[2]
            path = kg.find_path(source, target)
            if path:
                print(f"\n🛤️ 从 '{source}' 到 '{target}' 的路径:")
                for entity, relation in path:
                    if relation:
                        print(f"   [{entity.name}] --{relation.relation_type.value}-->")
                    else:
                        print(f"   [{entity.name}]")
            else:
                print(f"❌ 未找到路径")
        
        elif action == "neighbors" and len(parts) >= 2:
            name = " ".join(parts[1:])
            subgraph = kg.get_neighbors(name, hops=1)
            if subgraph.nodes:
                print(f"\n🔗 '{name}' 的邻居:")
                for rel in subgraph.edges:
                    print(f"   {rel.source} --[{rel.relation_type.value}]--> {rel.target}")
            else:
                print(f"❌ 未找到实体: {name}")
        
        elif action == "stats":
            stats = kg.get_statistics()
            print(f"\n📈 统计信息:")
            print(f"   实体: {stats['num_nodes']}")
            print(f"   关系: {stats['num_edges']}")
            print(f"   实体类型: {stats.get('entity_type_counts', {})}")
            print(f"   关系类型: {stats.get('relation_type_counts', {})}")
        
        else:
            print("❓ 未知命令。可用命令: entity, path, neighbors, stats, quit")
    
    store.close()
    print("\n👋 再见！")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🎯 Neo4j GraphRAG 演示")
    print("=" * 60)
    
    print("\n请选择演示模式:")
    print("1. Neo4j 图存储演示 (存储示例数据)")
    print("2. GraphRAG + Neo4j 演示 (图谱查询)")
    print("3. 交互式演示")
    print("4. 退出")
    
    while True:
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == "1":
            demo_neo4j_storage()
        elif choice == "2":
            demo_graph_rag_with_neo4j()
        elif choice == "3":
            demo_interactive()
        elif choice == "4":
            print("👋 再见！")
            break
        else:
            print("无效选项，请重新输入")


if __name__ == "__main__":
    main()
