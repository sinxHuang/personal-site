interface Friend {
  avatar: string; // 头像链接
  name: string; // 用户 id
  link: string; // 博客链接
  title?: string; // 用户头衔
  tag?: string; // 用户标签
  color?: string; // 标签颜色
}

/**
 * TODO: 缺项处理
 * 在此处填写你的友情链接
 */
export const friendsInfo: Friend[] = [
   {
     avatar: "https://avatars.githubusercontent.com/u/57825561?v=4",
     name: "Runze Cheng",
     title: "🐺 A wolf Developer",
     link: "https://aidcheng.github.io",
     tag: "UCL CS",
     color: "indigo",
   },
   {
     avatar: "https://avatars.githubusercontent.com/u/114977087?v=4",
     name: "Yuyang Song",
     title: "⚡️ LLM & Database",
     link: "https://yuyang-song.github.io",
     tag: "SCU CS",
     color: "indigo",
   },
   {
     avatar: "https://avatars.githubusercontent.com/u/141292166?v=4",
     name: "Sashimi-Balls",
     title: "📡 Wireless Communication",
     link: "https://sashimi-balls.github.io",
     tag: "UESTC CE",
     color: "sky",
   },
   {
     avatar: "https://avatars.githubusercontent.com/u/60817914?v=4",
     name: "Charles-Donne",
     title: "🤖 VLN & Agent & Robotics",
     link: "https://github.com/Charles-Donne",
     tag: "CSU EE",
     color: "sky",
   },
   {
     avatar: "https://avatars.githubusercontent.com/u/195511320?v=4",
     name: "ZHANGXiyuan",
     title: "🧩 Multimodal & AI4S",
     link: "https://github.com/ZHANGXiyuan2004",
     tag: "UESTC EE",
     color: "sky",
   },

];
