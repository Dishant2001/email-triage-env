from envs.my_env.client import MyEnv
from envs.my_env.models import MyAction

import asyncio


async def main():
    async with MyEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        print(result.observation.current_time)

        result = await client.step(MyAction(email_id="1", action_type="escalate"))
        print(result.observation.inbox)


if __name__ == "__main__":
    asyncio.run(main())