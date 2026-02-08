# =====================================================
# Hot News Module - Config Command
# =====================================================

import typer
from typing import Optional
import pymysql
from hot_news.config import get_settings

cmd_config = typer.Typer(help="配置管理")


@cmd_config.command("domain")
def config_domain(
    action: str = typer.Argument(..., help="操作: list/add/enable/disable/delete"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="领域名称"),
    keywords: Optional[str] = typer.Option(
        None, "--keywords", "-k", help="关键词，逗号分隔"
    ),
):
    """领域配置管理"""
    settings = get_settings()
    conn = pymysql.connect(**settings._db_config.connection_kwargs)

    if action == "list":
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM domain_config ORDER BY id")
            rows = cursor.fetchall()
            print("\n领域配置:")
            print("-" * 60)
            for row in rows:
                print(
                    f"  ID: {row[0]}, 名称: {row[1]}, 关键词: {row[2]}, 启用: {row[3]}"
                )
            print("-" * 60)

    elif action == "add":
        if not name or not keywords:
            print("[Error] 请指定 --name 和 --keywords")
            raise typer.Exit(1)
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO domain_config (domain_name, domain_keywords) VALUES (%s, %s)",
                (name, keywords),
            )
        conn.commit()
        print(f"[Success] 添加领域: {name}")

    elif action == "enable":
        if not name:
            print("[Error] 请指定 --name")
            raise typer.Exit(1)
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE domain_config SET is_enabled=1 WHERE domain_name=%s", (name,)
            )
        conn.commit()
        print(f"[Success] 启用领域: {name}")

    elif action == "disable":
        if not name:
            print("[Error] 请指定 --name")
            raise typer.Exit(1)
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE domain_config SET is_enabled=0 WHERE domain_name=%s", (name,)
            )
        conn.commit()
        print(f"[Success] 禁用领域: {name}")

    elif action == "delete":
        if not name:
            print("[Error] 请指定 --name")
            raise typer.Exit(1)
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM domain_config WHERE domain_name=%s", (name,))
        conn.commit()
        print(f"[Success] 删除领域: {name}")

    else:
        print(f"[Error] 未知操作: {action}")
        raise typer.Exit(1)

    conn.close()


@cmd_config.command("llm")
def config_llm(
    action: str = typer.Argument(..., help="操作: list/add/enable/disable/set-default"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="提供商: deepseek/qwen"
    ),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API密钥"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="模型名称"),
    api_base: Optional[str] = typer.Option(None, "--api-base", "-a", help="API地址"),
):
    """LLM配置管理"""
    settings = get_settings()
    conn = pymysql.connect(**settings._db_config.connection_kwargs)

    if action == "list":
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM llm_config ORDER BY id")
            rows = cursor.fetchall()
            print("\nLLM配置:")
            print("-" * 60)
            for row in rows:
                print(
                    f"  ID: {row[0]}, 提供商: {row[1]}, 模型: {row[4]}, 启用: {row[7]}, 默认: {row[8]}"
                )
            print("-" * 60)

    elif action == "add":
        if not provider or not api_key:
            print("[Error] 请指定 --provider 和 --api-key")
            raise typer.Exit(1)
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO llm_config (provider, api_base, api_key, model_name) VALUES (%s, %s, %s, %s)",
                (provider, api_base or "", api_key, model or ""),
            )
        conn.commit()
        print(f"[Success] 添加LLM配置: {provider}")

    elif action == "set-default":
        if not provider:
            print("[Error] 请指定 --provider")
            raise typer.Exit(1)
        with conn.cursor() as cursor:
            cursor.execute("UPDATE llm_config SET is_default=0")
            cursor.execute(
                "UPDATE llm_config SET is_default=1 WHERE provider=%s", (provider,)
            )
        conn.commit()
        print(f"[Success] 设置默认LLM: {provider}")

    elif action == "enable":
        if not provider:
            print("[Error] 请指定 --provider")
            raise typer.Exit(1)
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE llm_config SET is_enabled=1 WHERE provider=%s", (provider,)
            )
        conn.commit()
        print(f"[Success] 启用LLM: {provider}")

    elif action == "disable":
        if not provider:
            print("[Error] 请指定 --provider")
            raise typer.Exit(1)
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE llm_config SET is_enabled=0 WHERE provider=%s", (provider,)
            )
        conn.commit()
        print(f"[Success] 禁用LLM: {provider}")

    else:
        print(f"[Error] 未知操作: {action}")
        raise typer.Exit(1)

    conn.close()


@cmd_config.command("platform")
def config_platform(
    action: str = typer.Argument(..., help="操作: list/enable/disable"),
    codes: Optional[str] = typer.Option(
        None, "--codes", "-c", help="平台代码，逗号分隔"
    ),
):
    """热点平台配置管理"""
    settings = get_settings()
    conn = pymysql.connect(**settings._db_config.connection_kwargs)

    if action == "list":
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM hot_platform_config ORDER BY priority DESC")
            rows = cursor.fetchall()
            print("\n热点平台配置:")
            print("-" * 60)
            for row in rows:
                status = "✓ 启用" if row[3] else "✗ 禁用"
                print(f"  {row[1]:15} {row[2]:20} {status} 优先级:{row[4]}")
            print("-" * 60)

    elif action == "enable":
        if not codes:
            print("[Error] 请指定 --codes")
            raise typer.Exit(1)
        code_list = [c.strip() for c in codes.split(",")]
        with conn.cursor() as cursor:
            for code in code_list:
                cursor.execute(
                    "UPDATE hot_platform_config SET is_enabled=1 WHERE platform_code=%s",
                    (code,),
                )
        conn.commit()
        print(f"[Success] 启用平台: {codes}")

    elif action == "disable":
        if not codes:
            print("[Error] 请指定 --codes")
            raise typer.Exit(1)
        code_list = [c.strip() for c in codes.split(",")]
        with conn.cursor() as cursor:
            for code in code_list:
                cursor.execute(
                    "UPDATE hot_platform_config SET is_enabled=0 WHERE platform_code=%s",
                    (code,),
                )
        conn.commit()
        print(f"[Success] 禁用平台: {codes}")

    conn.close()
