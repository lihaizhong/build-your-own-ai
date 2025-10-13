async function main({ params }: Args): Promise<Output> {
    // 获取传入的参数
    const now_time = params.now_time;        // 例如 "3月2日" 或 "03月02日"
    const publish_time = params.publish_time; // 例如 "2025年03月02日 22:32"

    // 解析 now_time 月日
    function parseNowMonthDay(str) {
        // 匹配1或2位数字的月和日
        const match = str.match(/(\d{1,2})月(\d{1,2})日/);
        if (match) {
            return {
                month: parseInt(match[1], 10),
                day: parseInt(match[2], 10)
            };
        }
        return { month: null, day: null };
    }

    // 解析 publish_time 月日
    function parsePublishMonthDay(str) {
        // 假设格式总是 "YYYY年MM月DD日 HH:MM"
        const month = parseInt(str.slice(5, 7), 10);
        const day = parseInt(str.slice(8, 10), 10);
        return { month, day };
    }

    const now = parseNowMonthDay(now_time);
    const pub = parsePublishMonthDay(publish_time);

    // 判断月日是否相同
    const same_month_day = (now.month === pub.month && now.day === pub.day) ? 1 : 0;

    // 构建输出对象
    const ret = {
        same_month_day
    };

    return ret;